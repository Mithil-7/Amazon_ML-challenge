"""
Quick local sanity check of matching_results.tsv / candidate_pairs.tsv
BEFORE running the organizer-provided utils/validate_submission.py.

This is deliberately a light stdlib-only mirror of the documented rules:
  - one row per test Source-1 entity, no duplicates
  - id lists reference only S2-/S3- ids that exist in the test set
  - no duplicate ids within a single row's list
  - every matched id in matching_results also appears in candidate_pairs
"""
import argparse
import csv
from pathlib import Path


def load_ids(path):
    ids = set()
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids.add(row["entity_id"])
    return ids


def load_tsv_lists(path, list_col):
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"]
            raw = row.get(list_col, "") or ""
            ids = [x for x in raw.split(",") if x]
            out[s1] = ids
    return out


def check(test_dir, matching_path, candidate_path):
    issues = []
    test_dir = Path(test_dir)
    s1_ids = load_ids(test_dir / "test_source1.tsv")
    valid_other_ids = load_ids(test_dir / "test_source2.tsv") | \
        load_ids(test_dir / "test_source3.tsv")

    matches = load_tsv_lists(matching_path, "matched_entity_ids")
    cands = load_tsv_lists(candidate_path, "candidate_entity_ids")

    missing = s1_ids - set(matches)
    if missing:
        issues.append(f"{len(missing)} S1 entities missing from "
                       f"matching_results.tsv, e.g. {sorted(missing)[:3]}")
    missing_c = s1_ids - set(cands)
    if missing_c:
        issues.append(f"{len(missing_c)} S1 entities missing from "
                       f"candidate_pairs.tsv, e.g. {sorted(missing_c)[:3]}")

    for s1, ids in matches.items():
        if len(ids) != len(set(ids)):
            issues.append(f"{s1}: duplicate ids in matching_results row")
        bad = [i for i in ids if i not in valid_other_ids]
        if bad:
            issues.append(f"{s1}: matched ids not in test set: {bad}")
        cand_set = set(cands.get(s1, []))
        not_in_cand = [i for i in ids if i not in cand_set]
        if not_in_cand:
            issues.append(f"{s1}: matched ids not present in candidates: "
                           f"{not_in_cand}")

    for s1, ids in cands.items():
        if len(ids) != len(set(ids)):
            issues.append(f"{s1}: duplicate ids in candidate_pairs row")

    if issues:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for i in issues[:50]:
            print(" -", i)
    else:
        print("Local format check PASS "
              "(still run the organizer's utils/validate_submission.py).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--candidate", default="output/candidate_pairs.tsv")
    args = ap.parse_args()
    check(args.test_dir, args.matching, args.candidate)
