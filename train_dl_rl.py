"""
End-to-end trainer for the DL+RL architecture (replaces train.py's
LightGBM stage-2 + hand-tuned threshold with dl_matcher.py + rl_decision.py).

Split strategy: the held-out split is chosen over Source-1 ENTITY IDS
directly (not over candidate rows), so entities with zero blocking
candidates are still correctly assigned to train/val and correctly scored
as 0.0/1.0 in the final metric -- a gap in the earlier LightGBM baseline's
threshold tuning that this script fixes.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_source, load_ground_truth, ground_truth_to_pairs
from blocking import generate_candidates
from normalize import normalize_name, normalize_address
from dl_matcher import DLMatcher
from rl_decision import train_policy, f_beta, StoppingPolicy

AUX_COLS = ["name_cosine_sim", "country_match"]


def build_texts_and_country(*source_dfs):
    texts, country = {}, {}
    for df in source_dfs:
        for row in df.itertuples(index=False):
            texts[row.entity_id] = {
                "name": normalize_name(row.business_name),
                "addr": normalize_address(row.business_address),
            }
            country[row.entity_id] = (row.country or "").strip().lower()
    return texts, country


def add_aux_features(candidates: pd.DataFrame, country: dict) -> pd.DataFrame:
    candidates = candidates.copy()
    c1 = candidates["source1_entity_id"].map(country).fillna("")
    c2 = candidates["other_id"].map(country).fillna("")
    candidates["country_match"] = ((c1 == c2) & (c1 != "")).astype(np.float32)
    candidates["name_cosine_sim"] = candidates["name_cosine_sim"].astype(np.float32)
    return candidates


def sorted_dicts_by_s1(df: pd.DataFrame):
    probs_by_s1, ids_by_s1 = {}, {}
    for s1id, grp in df.groupby("source1_entity_id"):
        order = np.argsort(-grp["score"].to_numpy())
        probs_by_s1[s1id] = grp["score"].to_numpy()[order]
        ids_by_s1[s1id] = grp["other_id"].to_numpy()[order].tolist()
    return probs_by_s1, ids_by_s1


def full_macro_f05(s1_ids, probs_by_s1, ids_by_s1, true_by_s1, policy=None,
                    fixed_threshold=None):
    """Macro F0.5 over ALL s1_ids, including those with zero candidates
    (which contribute 1.0 if truly a singleton, else 0.0), matching the
    official scoring exactly."""
    scores = []
    for s1 in s1_ids:
        true_set = true_by_s1.get(s1, set())
        if s1 not in probs_by_s1:
            pred_set = set()
        else:
            probs, ids = probs_by_s1[s1], ids_by_s1[s1]
            if policy is not None:
                k, _ = policy.rollout(probs, greedy=True)
                pred_set = set(ids[:k])
            else:
                pred_set = {i for i, p in zip(ids, probs) if p >= fixed_threshold}
        scores.append(f_beta(pred_set, true_set))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../dataset/train")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--dl-epochs", type=int, default=10)
    ap.add_argument("--rl-epochs", type=int, default=20)
    ap.add_argument("--model-dir", default="../model_dl_rl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    true_pairs = ground_truth_to_pairs(gt)
    true_by_s1 = defaultdict(set)
    for a, b in true_pairs:
        true_by_s1[a].add(b)

    print(f"[train_dl_rl] {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3, "
          f"{len(true_pairs)} GT positive pairs")

    candidates = generate_candidates(s1, s2, s3)
    texts, country = build_texts_and_country(s1, s2, s3)
    candidates = add_aux_features(candidates, country)
    candidates["label"] = [
        1 if oid in true_by_s1.get(s1id, set()) else 0
        for s1id, oid in zip(candidates["source1_entity_id"], candidates["other_id"])
    ]
    print(f"[train_dl_rl] {len(candidates)} candidate pairs, "
          f"{candidates['label'].sum()} positive")

    # split by S1 ENTITY ID (not candidate row) so zero-candidate entities
    # are correctly bucketed too
    rng = np.random.default_rng(args.seed)
    all_s1_ids = s1["entity_id"].tolist()
    shuffled = rng.permutation(all_s1_ids)
    n_val = max(1, int(args.val_frac * len(shuffled)))
    val_s1_ids = set(shuffled[:n_val].tolist())
    train_s1_ids = set(shuffled[n_val:].tolist())

    train_df = candidates[candidates["source1_entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    val_df = candidates[candidates["source1_entity_id"].isin(val_s1_ids)].reset_index(drop=True)
    print(f"[train_dl_rl] split: {len(train_s1_ids)} train S1 / "
          f"{len(val_s1_ids)} val S1  "
          f"({len(train_df)} / {len(val_df)} candidate rows)")

    # ---- 1. DL matcher ----
    matcher = DLMatcher(aux_cols=AUX_COLS)
    matcher.fit(train_df, texts, train_df["label"].to_numpy(),
                epochs=args.dl_epochs)

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["score"] = matcher.predict_proba(train_df, texts)
    val_df["score"] = matcher.predict_proba(val_df, texts)

    train_probs, train_ids = sorted_dicts_by_s1(train_df)
    val_probs, val_ids = sorted_dicts_by_s1(val_df)

    # ---- baseline: fixed-threshold decision (no RL), for comparison ----
    best_fixed_t, best_fixed_f = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        f = full_macro_f05(val_s1_ids, val_probs, val_ids, true_by_s1,
                            fixed_threshold=t)
        if f > best_fixed_f:
            best_fixed_f, best_fixed_t = f, t
    print(f"[train_dl_rl] baseline fixed-threshold={best_fixed_t:.2f} -> "
          f"val macro F0.5={best_fixed_f:.4f} (no RL)")

    # ---- 2. RL stopping policy, trained on the TRAIN fold's matcher scores ----
    policy = train_policy(train_probs, true_by_s1, train_ids,
                           epochs=args.rl_epochs, seed=args.seed)

    rl_f = full_macro_f05(val_s1_ids, val_probs, val_ids, true_by_s1, policy=policy)
    print(f"[train_dl_rl] RL policy -> val macro F0.5={rl_f:.4f} "
          f"(delta vs fixed threshold: {rl_f - best_fixed_f:+.4f})")

    # ---- save everything predict_dl_rl.py needs ----
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    matcher.save(model_dir / "dl_matcher.pkl")
    import pickle
    with open(model_dir / "rl_policy.pkl", "wb") as f:
        pickle.dump(policy.save_state(), f)
    with open(model_dir / "report.json", "w") as f:
        import json
        json.dump({
            "n_train_s1": len(train_s1_ids), "n_val_s1": len(val_s1_ids),
            "n_candidate_pairs": int(len(candidates)),
            "baseline_fixed_threshold": float(best_fixed_t),
            "baseline_fixed_threshold_f05": float(best_fixed_f),
            "rl_policy_f05": float(rl_f),
        }, f, indent=2)
    print(f"[train_dl_rl] saved DL matcher + RL policy to {model_dir}/")


if __name__ == "__main__":
    main()
