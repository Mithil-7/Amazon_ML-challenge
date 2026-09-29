"""
Decision-rule post-processing, applied on top of a model's ALREADY-SCORED
candidates (from train.py's val_scored.tsv or predict.py's
scored_candidates.tsv). This is deliberately separated from train.py /
predict.py so you can experiment with decision rules in seconds instead of
re-running blocking + features + the model every time.

Two rules, both named directly in the reference architecture's table:

1. ONE-TO-ONE ENFORCEMENT: if a Source-2/3 id is a plausible candidate for
   several Source-1 entities, keep it only for the S1 it scores highest
   against. This assumes one_to_one_holds in your ground truth (check your
   stats.json / the ground truth file for this before trusting it -- if a
   single S2/S3 id can legitimately match multiple S1 entities in your
   data, this rule will silently and incorrectly delete some true matches).

2. EXPECTED-F0.5 SELECTION: instead of a single global probability
   threshold, sort each S1's candidates by score and pick the prefix
   length m that maximizes an estimate of that entity's own F0.5, using
   only the candidate probabilities (no ground truth needed at test time):
     for m >= 1:  E[F] ~= 1.25 * sum(p_1..p_m) / (m + 0.25*(sum(p_1..p_n) + lambda))
     for m = 0 :  E[F] ~= prod(1 - p_i) * exp(-lambda)
   lambda represents "expected missed-match mass" and is tuned on your val
   cache by grid search to maximize the REAL macro F0.5 (not just the
   proxy above) -- this is the exact formula from the original task brief.

Usage:
    # tune lambda + compare variants on your val cache (has true labels)
    python3 postprocess.py --scored ../model/val_scored.tsv --mode val

    # apply the chosen rule to your test cache (no labels) and write the
    # final submission files
    python3 postprocess.py --scored ../output/scored_candidates.tsv \
        --mode test --lambda-val 0.4 --rule one_to_one_expected_f05 \
        --test-dir ../dataset/test --out-dir ../output
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_source, write_id_list_tsv


def f_beta(pred_ids: set, true_ids: set, beta=0.5) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_ids)
    rec = tp / len(true_ids)
    b2 = beta ** 2
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def enforce_one_to_one(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each other_id, only the row with the highest score across
    all competing S1 entities. Vectorized (idxmax per group), not a
    per-group Python loop -- safe at millions of rows."""
    idx = df.groupby("other_id")["score"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def expected_f05_select(group_scores: np.ndarray, lam: float) -> int:
    """Return m (0..len(group_scores)), the prefix length that maximizes
    the expected-F0.5 estimate for one S1 entity's sorted (desc) scores."""
    p = np.clip(np.sort(group_scores)[::-1], 1e-6, 1 - 1e-6)
    n = len(p)
    total = p.sum()
    best_m, best_e = 0, float(np.prod(1 - p)) * np.exp(-lam)
    running = 0.0
    for m in range(1, n + 1):
        running += p[m - 1]
        e = 1.25 * running / (m + 0.25 * (total + lam))
        if e > best_e:
            best_e, best_m = e, m
    return best_m


def apply_expected_f05(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """df must be sorted by ['source1_entity_id','score' desc] already is
    NOT required -- this sorts internally per group. Returns only the
    KEPT rows."""
    kept_rows = []
    for s1_id, grp in df.groupby("source1_entity_id", sort=False):
        grp_sorted = grp.sort_values("score", ascending=False)
        scores = grp_sorted["score"].to_numpy()
        m = expected_f05_select(scores, lam)
        if m > 0:
            kept_rows.append(grp_sorted.iloc[:m])
    if not kept_rows:
        return df.iloc[0:0]
    return pd.concat(kept_rows, ignore_index=True)


def macro_f05_report(pred_df: pd.DataFrame, true_by_s1: dict, all_s1_ids,
                      country_by_s1: dict = None):
    pred_by_s1 = {s1: set(g["other_id"]) for s1, g in
                  pred_df.groupby("source1_entity_id")}
    scores, by_country = [], {}
    n_singleton_correct, n_singleton_total = 0, 0
    for s1 in all_s1_ids:
        true_set = true_by_s1.get(s1, set())
        pred_set = pred_by_s1.get(s1, set())
        f = f_beta(pred_set, true_set)
        scores.append(f)
        if not true_set:
            n_singleton_total += 1
            if not pred_set:
                n_singleton_correct += 1
        if country_by_s1:
            c = country_by_s1.get(s1, "?")
            by_country.setdefault(c, []).append(f)

    report = {
        "macro_f05": float(np.mean(scores)),
        "n_s1": len(scores),
        "singleton_accuracy": (n_singleton_correct / n_singleton_total
                                 if n_singleton_total else None),
        "n_singletons": n_singleton_total,
    }
    if by_country:
        report["by_country"] = {c: float(np.mean(v)) for c, v in by_country.items()}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True,
                     help="path to val_scored.tsv (mode=val, has a 'label' "
                          "column) or scored_candidates.tsv (mode=test, no labels)")
    ap.add_argument("--mode", choices=["val", "test"], required=True)
    ap.add_argument("--test-dir", default=None,
                     help="required for mode=test, to get the full S1 id list")
    ap.add_argument("--out-dir", default="../output")
    ap.add_argument("--lambda-grid", default="0.0,0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0")
    ap.add_argument("--lambda-val", type=float, default=None,
                     help="skip the grid search and use this lambda directly "
                          "(mode=test always requires this, or --rule fixed_threshold)")
    ap.add_argument("--rule", default=None,
                     choices=["threshold", "one_to_one_threshold",
                              "expected_f05", "one_to_one_expected_f05"],
                     help="mode=test: which rule to apply. mode=val: if "
                          "omitted, ALL FOUR are compared and reported.")
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="used by the threshold-based rules")
    args = ap.parse_args()

    df = pd.read_csv(args.scored, sep="\t", dtype={"source1_entity_id": str,
                                                       "other_id": str, "source": str})
    print(f"[postprocess] loaded {len(df)} scored candidate rows")

    if args.mode == "val":
        if "label" not in df.columns:
            raise SystemExit("--mode val requires a 'label' column "
                              "(use train.py's val_scored.tsv)")
        true_by_s1 = {}
        for s1, oid in zip(df.loc[df["label"] == 1, "source1_entity_id"],
                            df.loc[df["label"] == 1, "other_id"]):
            true_by_s1.setdefault(s1, set()).add(oid)
        all_s1_ids = df["source1_entity_id"].unique().tolist()

        variants = {
            "threshold": lambda d: d[d["score"] >= args.threshold],
            "one_to_one_threshold": lambda d: enforce_one_to_one(d[d["score"] >= args.threshold]),
        }

        lambdas = [float(x) for x in args.lambda_grid.split(",")]
        best_lam_plain, best_f_plain = None, -1.0
        best_lam_o2o, best_f_o2o = None, -1.0
        for lam in lambdas:
            pred_plain = apply_expected_f05(df, lam)
            f_plain = macro_f05_report(pred_plain, true_by_s1, all_s1_ids)["macro_f05"]
            if f_plain > best_f_plain:
                best_f_plain, best_lam_plain = f_plain, lam

            pred_o2o = apply_expected_f05(enforce_one_to_one(df), lam)
            f_o2o = macro_f05_report(pred_o2o, true_by_s1, all_s1_ids)["macro_f05"]
            if f_o2o > best_f_o2o:
                best_f_o2o, best_lam_o2o = f_o2o, lam
        print(f"[postprocess] expected-F0.5 lambda grid -> best plain "
              f"lambda={best_lam_plain} (val F0.5={best_f_plain:.4f}), "
              f"best one-to-one lambda={best_lam_o2o} (val F0.5={best_f_o2o:.4f})")

        variants["expected_f05"] = lambda d: apply_expected_f05(d, best_lam_plain)
        variants["one_to_one_expected_f05"] = lambda d: apply_expected_f05(
            enforce_one_to_one(d), best_lam_o2o)

        print("\n[postprocess] comparing all 4 decision rules on val:")
        results = {}
        for name, fn in variants.items():
            pred = fn(df)
            report = macro_f05_report(pred, true_by_s1, all_s1_ids)
            results[name] = report
            print(f"  {name:28s} macro F0.5={report['macro_f05']:.4f}  "
                  f"singleton_acc={report['singleton_accuracy']}")

        best_name = max(results, key=lambda k: results[k]["macro_f05"])
        print(f"\n[postprocess] BEST on val: {best_name} "
              f"(F0.5={results[best_name]['macro_f05']:.4f}). "
              f"Re-run with --mode test --rule {best_name}"
              + (f" --lambda-val {best_lam_o2o if 'one_to_one' in best_name else best_lam_plain}"
                 if "expected_f05" in best_name else f" --threshold {args.threshold}"))

    else:  # mode == test
        if args.test_dir is None:
            raise SystemExit("--mode test requires --test-dir")
        if args.rule is None:
            raise SystemExit("--mode test requires --rule (see the val run's "
                              "recommendation for which one to pick)")
        test_dir = Path(args.test_dir)
        s1 = load_source(test_dir / "test_source1.tsv")
        all_s1_ids = s1["entity_id"].tolist()

        if args.rule == "threshold":
            pred = df[df["score"] >= args.threshold]
        elif args.rule == "one_to_one_threshold":
            pred = enforce_one_to_one(df[df["score"] >= args.threshold])
        elif args.rule == "expected_f05":
            if args.lambda_val is None:
                raise SystemExit("--rule expected_f05 requires --lambda-val")
            pred = apply_expected_f05(df, args.lambda_val)
        else:  # one_to_one_expected_f05
            if args.lambda_val is None:
                raise SystemExit("--rule one_to_one_expected_f05 requires --lambda-val")
            pred = apply_expected_f05(enforce_one_to_one(df), args.lambda_val)

        out_dir = Path(args.out_dir)
        write_id_list_tsv(pred[["source1_entity_id", "other_id"]], all_s1_ids,
                           out_dir / "matching_results.tsv", "matched_entity_ids")
        print(f"[postprocess] rule={args.rule} -> {len(pred)} predicted "
              f"match pairs across {pred['source1_entity_id'].nunique()} S1 "
              f"entities. Wrote {out_dir/'matching_results.tsv'}")
        print("[postprocess] NOTE: candidate_pairs.tsv is unchanged from "
              "predict.py's original output -- this script only rewrites "
              "matching_results.tsv.")


if __name__ == "__main__":
    main()
