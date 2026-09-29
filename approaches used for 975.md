# Approaches and results

Metric: macro F0.5 over all Source-1 entities.
- **Val**: 441,364 held-out S1.
- **Sim val**: the same S1 in a test-like population (413k unused train S1 removed, so ~40% of DB records are unowned, as in test).
- **LB**: leaderboard.

## Final architecture (B9)

| stage | component |
|---|---|
| Cleaning | Unicode/Indic transliteration, legal-suffix canonicalisation, address parsing, learned abbreviation maps, French rules (R→rue, ch→chemin, q→quai, department→region) |
| Blocking | multilingual-e5-small kNN per source within country: name+address k=10, name k=5, name over address-less records k=5, address-only k=3 (on-the-fly); plus 14 exact keys |
| Pruner | LightGBM on cheap features: top-10 & p1 ≥ 0.01 → 5.6 candidates/S1, pair recall 0.974 |
| Train population | test-like (413k S1 removed) |
| Matcher | LightGBM (~160 features: string, address, house number, context, competition, noise-invariant name features) + e5-small cross-encoder + xlm-roberta-base cross-encoder (0.05 ≤ p1 < 0.9) |
| Stacking | 4-fold OOF LightGBM with sibling and S2/S3 agreement features |
| Decision | one-to-one + expected-F0.5 per S1 |

## Main versions

| version | architecture | blocking recall | val | sim val | LB |
|---|---|---|---|---|---|
| B0 | name kNN (k=10) + 9 exact keys → LightGBM pruner → LightGBM matcher → threshold rule | 0.795 | 0.8879 | | |
| B1 | + name+address kNN (k=10), name kNN k=5, 5 address keys (14 keys) | 0.9666 | 0.97435 | | 0.973 |
| B2 | B1 + house-number numeric features + e5-small cross-encoder feature | 0.9666 | 0.98006 | | |
| B4 | B2 + name kNN over address-less records (k=5) + address-only kNN (k=3) + token-alignment / postcode features | 0.9773 | 0.98185 | 0.98117 | 0.975 |
| B4-FR | B4 + French normalisation on test | 0.9773 | 0.98185 | 0.98117 | |
| B6 | B4 matcher retrained on the test-like population | 0.9773 | | 0.98149 | |
| B7 | B6 + noise-invariant name features | 0.9773 | | 0.98158 | |
| B8 | B7 + xlm-roberta-base cross-encoder (p1 < 0.98) | 0.9773 | | 0.98201 | |
| B8g | B7 + xlm-roberta-base cross-encoder (0.05 ≤ p1 < 0.9) | 0.9773 | | 0.98191 | |
| **B9 (final)** | **B8g + stacking with S2/S3 agreement features** | **0.9773** | | **0.98250** | |

## Other experiments

| experiment | architecture | result |
|---|---|---|
| Stacking on B1 | 4-fold OOF + sibling probability features | val 0.97573 (+0.0014) |
| Stacking on B2 | same | val 0.98065 (+0.0006) |
| Stacking + agreement on B2 | + S2/S3 agreement features | val 0.98076 (+0.0007) |
| Stacking + agreement on B4 | same | val 0.98257 (+0.0007) |
| 3-seed ensemble on B2 | 3 × LightGBM averaged | val 0.98012 (+0.0001) |
| XGBoost on B2 features | XGBoost hist, 127 leaves | val 0.98009 |
| LightGBM + XGBoost blend | average | val 0.98012 (+0.0001) |
| CE v2 on B2 (B3) | second e5-small CE, hard negatives from top-15 | val 0.98051 (+0.0005) |
| xlm-r CE on B4 (B5) | xlm-roberta-base CE, p1 < 0.98 | val 0.98230 (+0.0005) |
| One-to-one after threshold | decision variant | +0.0000 |
| Per-bucket thresholds | threshold per #candidates × source | +0.0001 |
| Isotonic + expected-F0.5 | calibrated per-S1 set selection | −0.0001 |
| Refit proxy | + 55% training S1 | +0.0005 |
| 3.7× training S1 (no CE) | matcher on G + 1.1M extra S1 | no gain (logloss 0.04889 vs 0.04886) |
| Canonical + phonetic name keys | extra exact blocking keys (cap 50) | recall +0.0006 |
| Blocking k grid | name+addr k / name k: 5/0, 7/3, 10/5, 20/10 | recall 0.956, 0.962, 0.966, 0.972 |
| Address-only kNN k grid (India) | k = 1, 3, 5, 10 | India recall 0.9531, 0.9560, 0.9575, 0.9592 |
