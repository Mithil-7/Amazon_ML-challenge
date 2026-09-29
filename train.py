"""
Train the pair-matching classifier.

Pipeline:
  1. Load train source1/2/3 + ground truth.
  2. Run the SAME blocking stage used at inference time to get candidate
     pairs for every train Source-1 entity (this is important: training on
     candidates the blocking stage would actually produce, not on all
     ground-truth pairs, keeps train/inference distributions consistent).
  3. Label each candidate pair 1 if it's a true match (in ground truth),
     else 0. Report the blocking recall ceiling (fraction of ground-truth
     positive pairs that made it into the candidate set) as a diagnostic.
  4. Split by Source-1 entity (group split) into train/val so no entity's
     pairs leak across the split.
  5. Compute pairwise features and train a gradient-boosted classifier
     (LightGBM if available, else sklearn GradientBoostingClassifier as a
     dependency-light fallback -- both are permissively licensed and have
     zero parameters in the "pretrained model" sense, so the 8B-param /
     MIT-Apache license constraint is a non-issue).
  6. Sweep the decision threshold on the validation split and pick the one
     that maximizes macro-averaged F0.5 across Source-1 entities
     (including singletons, exactly as it is scored on the leaderboard).
  7. Persist the model + threshold + feature column order to disk.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_source, load_ground_truth, ground_truth_to_pairs
from blocking import generate_candidates
from features import compute_features, build_record_lookup, FEATURE_COLS

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_LGBM = False


def f_beta_per_entity(pred_map: dict, true_map: dict, s1_ids, beta=0.5):
    """Macro-average F_beta over s1_ids, matching the leaderboard formula
    exactly (singletons: 1.0 if correctly predicted empty, else 0.0)."""
    scores = []
    for s1 in s1_ids:
        pred = pred_map.get(s1, set())
        true = true_map.get(s1, set())
        if not true and not pred:
            scores.append(1.0)
            continue
        if not pred:
            scores.append(0.0)
            continue
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        beta2 = beta ** 2
        denom = (beta2 * precision + recall)
        f = ((1 + beta2) * precision * recall / denom) if denom > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(val_df: pd.DataFrame, val_s1_ids, true_pairs: set):
    true_map = {}
    for s1, other in true_pairs:
        true_map.setdefault(s1, set()).add(other)

    # coarser grid (0.02 step = 46 thresholds instead of 91) and a vectorized
    # groupby instead of a per-row Python loop inside the sweep -- at
    # millions of val rows, doing the old per-row .setdefault() loop 91
    # times over was itself a meaningful slowdown, not just the O(n*m)
    # recall-ceiling bug below
    best_t, best_f = 0.5, -1.0
    small = val_df[["source1_entity_id", "other_id", "score"]]
    for t in np.arange(0.05, 0.96, 0.02):
        kept = small[small["score"] >= t]
        pred_map = kept.groupby("source1_entity_id")["other_id"].apply(set).to_dict()
        f = f_beta_per_entity(pred_map, true_map, val_s1_ids)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-out", default="model/matcher.pkl")
    ap.add_argument("--sample-s1", type=int, default=None,
                     help="Randomly subsample this many S1 entities (and "
                          "only their candidates/ground truth) before "
                          "featurizing/training. Use this to get a FAST "
                          "first working run (e.g. --sample-s1 50000) on "
                          "the full-size dataset while you wait on a full "
                          "run in the background.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t_start = time.time()

    train_dir = Path(args.train_dir)
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    true_pairs = ground_truth_to_pairs(gt)
    print(f"[train] loaded {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 records, "
          f"{len(true_pairs)} ground-truth positive pairs "
          f"({time.time()-t_start:.1f}s)")

    if args.sample_s1 is not None and args.sample_s1 < len(s1):
        rng = np.random.default_rng(args.seed)
        keep_ids = set(rng.choice(s1["entity_id"].to_numpy(),
                                    size=args.sample_s1, replace=False).tolist())
        s1 = s1[s1["entity_id"].isin(keep_ids)].reset_index(drop=True)
        true_pairs = {(a, b) for a, b in true_pairs if a in keep_ids}
        print(f"[train] --sample-s1 active: subsampled to {len(s1)} S1 "
              f"entities and {len(true_pairs)} ground-truth pairs. "
              f"NOTE: this is for a fast first pass only -- re-run without "
              f"--sample-s1 for your real submission once you have time.")

    candidates = generate_candidates(s1, s2, s3)
    print(f"[train] blocking produced {len(candidates)} candidate pairs "
          f"({time.time()-t_start:.1f}s elapsed)")

    # vectorized recall-ceiling check (set intersection), NOT a per-ground-
    # truth-pair table scan over candidates -- the old version was an
    # O(len(true_pairs) * len(candidates)) loop, which at 7.6M x tens of
    # millions would itself have been effectively infinite
    candidate_pair_set = set(zip(candidates["source1_entity_id"],
                                   candidates["other_id"]))
    recovered = len(true_pairs & candidate_pair_set)
    ceiling = recovered / len(true_pairs) if true_pairs else 1.0
    print(f"[train] blocking recall ceiling: {ceiling:.4f} "
          f"({recovered}/{len(true_pairs)} positives recovered) "
          f"({time.time()-t_start:.1f}s elapsed)")

    label_set = true_pairs
    candidates["label"] = [
        1 if (s1id, oid) in label_set else 0
        for s1id, oid in zip(candidates["source1_entity_id"],
                              candidates["other_id"])
    ]

    lookup_s1 = build_record_lookup(s1)
    lookup_other = build_record_lookup(pd.concat([s2, s3], ignore_index=True))
    feat_df = compute_features(candidates, lookup_s1, lookup_other)
    print(f"[train] feature computation done ({time.time()-t_start:.1f}s elapsed)")

    # group split by source1_entity_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(
        gss.split(feat_df, groups=feat_df["source1_entity_id"]))
    train_df, val_df = feat_df.iloc[train_idx], feat_df.iloc[val_idx]

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_val = val_df[FEATURE_COLS]

    if _HAS_LGBM:
        model = LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            min_child_samples=5, class_weight="balanced",
            random_state=42, verbosity=-1)
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42)
    model.fit(X_train, y_train)
    print(f"[train] fit {'LightGBM' if _HAS_LGBM else 'GradientBoosting'} "
          f"classifier on {len(X_train)} pairs "
          f"({y_train.sum()} positive / {len(y_train)-y_train.sum()} negative) "
          f"({time.time()-t_start:.1f}s elapsed)")

    val_df = val_df.copy()
    val_df["score"] = model.predict_proba(X_val)[:, 1]
    val_s1_ids = val_df["source1_entity_id"].unique().tolist()
    best_t, best_f = tune_threshold(val_df, val_s1_ids, true_pairs)
    print(f"[train] best threshold={best_t:.2f}  val macro-F0.5={best_f:.4f} "
          f"({time.time()-t_start:.1f}s elapsed)")

    # cache the val split's scored candidates + true labels to disk so
    # postprocess.py can experiment with different decision rules (one-to-
    # one enforcement, relative thresholds, ensembling, ...) WITHOUT paying
    # the cost of re-running blocking + features + training every time --
    # that's the expensive part at your real data scale.
    val_cache_path = Path(args.model_out).parent / "val_scored.tsv"
    val_cache_path.parent.mkdir(parents=True, exist_ok=True)
    val_df[["source1_entity_id", "other_id", "source", "score", "label"]].to_csv(
        val_cache_path, sep="\t", index=False)
    print(f"[train] cached val scored candidates to {val_cache_path} "
          f"({len(val_df)} rows) -- use this with postprocess.py to tune "
          f"decision rules quickly")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "threshold": float(best_t),
            "feature_cols": FEATURE_COLS,
            "has_lgbm": _HAS_LGBM,
        }, f)
    with open(Path(args.model_out).with_suffix(".meta.json"), "w") as f:
        json.dump({
            "threshold": float(best_t),
            "val_macro_f0.5": float(best_f),
            "blocking_recall_ceiling": float(ceiling),
            "n_candidate_pairs": int(len(candidates)),
        }, f, indent=2)
    print(f"[train] model saved to {args.model_out}")


if __name__ == "__main__":
    main()
