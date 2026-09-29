# Business Entity Resolution — Pipeline

End-to-end pipeline: country-aware blocking → TF-IDF/token candidate
generation → pairwise similarity features → gradient-boosted classifier →
F0.5-tuned threshold → submission files.

## Setup

```bash
pip install -r requirements.txt
```

## Directory layout expected

```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```
(Same layout as the `student_resource/` folder from the challenge; point
`--train-dir` / `--test-dir` elsewhere if yours differs.)

## Run end-to-end

```bash
cd src

# 1. Train the matcher (runs blocking internally on the train set,
#    labels candidate pairs from train_ground_truth.tsv, fits the
#    classifier, and tunes the decision threshold for macro F0.5).
python3 train.py --train-dir ../dataset/train --model-out ../model/matcher.pkl

# 2. Score the test set: runs the SAME blocking stage on test data,
#    scores every candidate with the trained model, and writes both
#    required output files.
python3 predict.py --test-dir ../dataset/test --model ../model/matcher.pkl --out-dir ../output

# 3. (optional) quick local format sanity check before the organizer's
#    official validator.
python3 check_format.py --test-dir ../dataset/test \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv

# 4. Run the organizer-provided validator from the student_resource/ dir:
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Outputs land in `output/matching_results.tsv` and `output/candidate_pairs.tsv`,
matching the exact schema required by the challenge (one row per test
Source-1 entity, comma-separated match/candidate lists, empty string for
singletons/no-candidates).

## Method summary

**Blocking / candidate generation** (`src/blocking.py`)
- Records are grouped into blocks by normalized `country` (open-set string
  match — an unseen label like `France` in the test set just forms its own
  block automatically, nothing is hardcoded to `{US, India}`).
- Within each country block, a character n-gram (2–4) TF-IDF vectorizer is
  fit jointly on normalized Source-1 and Source-2/3 names, and cosine
  `NearestNeighbors` retrieves the top-K (15) most name-similar
  Source-2/3 records per Source-1 record, subject to a minimum similarity
  floor.
- A secondary "rare shared token" pass adds any record sharing a
  distinctive (len ≥ 4, not overly common in the block) name token, to
  protect recall against typos concentrated in a small part of the name.
- The union is capped at 10 candidates per (Source-1, Source-2) and
  (Source-1, Source-3) each, keeping the highest-similarity ones — this is
  the exact candidate set written to `candidate_pairs.tsv` and fed to the
  matching model.

**Feature engineering** (`src/features.py`)
- Name: TF-IDF cosine similarity (from blocking), RapidFuzz token-sort
  ratio, RapidFuzz partial ratio, token-set Jaccard, length difference.
- Address: same four metrics after abbreviation normalization (Rd/Road,
  St/Street, ...) and landmark-phrase stripping (near/opposite/...).
- Country: exact-match binary flag.

**Matching model** (`src/train.py`)
- LightGBM classifier (falls back to scikit-learn
  `GradientBoostingClassifier` if LightGBM is unavailable) trained on
  candidate pairs from the blocking stage, labeled against
  `train_ground_truth.tsv`. Both options are permissively licensed
  (MIT/BSD) and are not pretrained/parameterized language models, so they
  satisfy the ≤8B-parameter / MIT-Apache constraint trivially.
- Train/validation split is grouped by `source1_entity_id` so no entity's
  pairs leak across the split.
- The decision threshold is swept on the validation split and chosen to
  maximize the exact macro-averaged F0.5 metric used on the leaderboard
  (singletons included, scored 1.0 when correctly predicted empty).

**No external data.** Only the provided train/test files are used — no
external lookups, geocoding, or pretrained embedding APIs, per the fair-play
rules.

## Files

```
src/
  normalize.py       # text normalization (names, addresses, country)
  io_utils.py        # tsv IO + submission-file writer
  blocking.py         # candidate generation (blocking stage)
  features.py         # pairwise similarity feature engineering
  train.py             # trains classifier + tunes F0.5 threshold
  predict.py           # produces candidate_pairs.tsv + matching_results.tsv
  check_format.py     # quick local format sanity check
```
