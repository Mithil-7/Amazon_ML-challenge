"""
Generate output/candidate_pairs.tsv and output/matching_results.tsv for the
test set, using the blocking stage + trained classifier + tuned threshold.

candidate_pairs.tsv = the full candidate set the model scores over (the
final blocking-stage output, per the spec).
matching_results.tsv = candidates whose model score >= tuned threshold.

Every Source-1 test entity gets exactly one row in both files, even when it
has zero candidates / matches (empty string).
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_source, write_id_list_tsv
from blocking import generate_candidates
from features import compute_features, build_record_lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model", default="model/matcher.pkl")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    print(f"[predict] {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 test records")

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model, threshold, feature_cols = (
        bundle["model"], bundle["threshold"], bundle["feature_cols"])

    candidates = generate_candidates(s1, s2, s3)
    print(f"[predict] blocking produced {len(candidates)} candidate pairs "
          f"for {candidates['source1_entity_id'].nunique()} / {len(s1)} "
          f"S1 entities")

    lookup_s1 = build_record_lookup(s1)
    lookup_other = build_record_lookup(pd.concat([s2, s3], ignore_index=True))
    feat_df = compute_features(candidates, lookup_s1, lookup_other)

    if len(feat_df):
        feat_df["score"] = model.predict_proba(feat_df[feature_cols])[:, 1]
    else:
        feat_df["score"] = pd.Series(dtype=float)

    out_dir = Path(args.out_dir)
    all_s1_ids = s1["entity_id"].tolist()

    cand_out = candidates.rename(columns={"other_id": "other_id"})[
        ["source1_entity_id", "other_id"]]
    write_id_list_tsv(cand_out, all_s1_ids, out_dir / "candidate_pairs.tsv",
                       "candidate_entity_ids")

    matched = feat_df[feat_df["score"] >= threshold][
        ["source1_entity_id", "other_id"]]
    write_id_list_tsv(matched, all_s1_ids, out_dir / "matching_results.tsv",
                       "matched_entity_ids")

    # cache ALL scored candidates (not just the ones that passed the fixed
    # threshold) so postprocess.py can apply one-to-one enforcement and/or
    # the expected-F0.5 decision rule WITHOUT re-running blocking + features
    # + the model on the full test set again -- that's the expensive part.
    scored_path = out_dir / "scored_candidates.tsv"
    feat_df[["source1_entity_id", "other_id", "source", "score"]].to_csv(
        scored_path, sep="\t", index=False)
    print(f"[predict] cached all scored candidates to {scored_path} "
          f"({len(feat_df)} rows) for use with postprocess.py")

    print(f"[predict] threshold={threshold:.2f} -> "
          f"{len(matched)} predicted match pairs across "
          f"{matched['source1_entity_id'].nunique()} S1 entities")
    print(f"[predict] wrote {out_dir/'candidate_pairs.tsv'} and "
          f"{out_dir/'matching_results.tsv'}")


if __name__ == "__main__":
    main()
