"""
Test-time inference for the DL+RL architecture. Mirrors predict.py's
contract exactly (same two output files, same one-row-per-S1 guarantee)
but scores candidates with the trained DLMatcher and decides cutoffs with
the trained RL StoppingPolicy instead of a LightGBM probability + fixed
threshold.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_source, write_id_list_tsv
from blocking import generate_candidates
from dl_matcher import DLMatcher
from rl_decision import StoppingPolicy
from train_dl_rl import build_texts_and_country, add_aux_features, sorted_dicts_by_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../dataset/test")
    ap.add_argument("--model-dir", default="../model_dl_rl")
    ap.add_argument("--out-dir", default="../output_dl_rl")
    ap.add_argument("--use-rl", action="store_true", default=True)
    ap.add_argument("--no-rl", dest="use_rl", action="store_false",
                     help="use the fixed-threshold fallback instead of the RL policy")
    ap.add_argument("--fixed-threshold", type=float, default=0.5)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    print(f"[predict_dl_rl] {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 test records")

    model_dir = Path(args.model_dir)
    matcher = DLMatcher.load(model_dir / "dl_matcher.pkl")
    policy = None
    if args.use_rl:
        policy = StoppingPolicy()
        with open(model_dir / "rl_policy.pkl", "rb") as f:
            policy.load_state(pickle.load(f))

    candidates = generate_candidates(s1, s2, s3)
    print(f"[predict_dl_rl] blocking produced {len(candidates)} candidate pairs "
          f"for {candidates['source1_entity_id'].nunique()} / {len(s1)} S1 entities")

    texts, country = build_texts_and_country(s1, s2, s3)
    candidates = add_aux_features(candidates, country)
    candidates["score"] = matcher.predict_proba(candidates, texts) if len(candidates) else []

    probs_by_s1, ids_by_s1 = sorted_dicts_by_s1(candidates) if len(candidates) else ({}, {})

    all_s1_ids = s1["entity_id"].tolist()
    match_rows = []
    for s1id in all_s1_ids:
        if s1id not in probs_by_s1:
            continue
        probs, ids = probs_by_s1[s1id], ids_by_s1[s1id]
        if policy is not None:
            k, _ = policy.rollout(probs, greedy=True)
            keep = ids[:k]
        else:
            keep = [i for i, p in zip(ids, probs) if p >= args.fixed_threshold]
        for other_id in keep:
            match_rows.append((s1id, other_id))

    matched_df = pd.DataFrame(match_rows, columns=["source1_entity_id", "other_id"])
    cand_df = candidates[["source1_entity_id", "other_id"]] if len(candidates) else \
        pd.DataFrame(columns=["source1_entity_id", "other_id"])

    out_dir = Path(args.out_dir)
    write_id_list_tsv(cand_df, all_s1_ids, out_dir / "candidate_pairs.tsv",
                       "candidate_entity_ids")
    write_id_list_tsv(matched_df, all_s1_ids, out_dir / "matching_results.tsv",
                       "matched_entity_ids")

    decision_name = "RL policy" if policy is not None else \
        f"fixed threshold={args.fixed_threshold}"
    print(f"[predict_dl_rl] decision rule: {decision_name}")
    print(f"[predict_dl_rl] {len(matched_df)} predicted match pairs across "
          f"{matched_df['source1_entity_id'].nunique() if len(matched_df) else 0} S1 entities")
    print(f"[predict_dl_rl] wrote {out_dir/'candidate_pairs.tsv'} and "
          f"{out_dir/'matching_results.tsv'}")


if __name__ == "__main__":
    main()
