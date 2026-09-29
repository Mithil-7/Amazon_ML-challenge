# Architecture 2: Gated Dual-DAN Matcher + RL Stopping Policy

This is a second, non-tree-based matching pipeline, offered alongside the
original LightGBM baseline (`train.py` / `predict.py`). It reuses the same
blocking stage (`blocking.py`) and I/O contract, but replaces:

- LightGBM binary classifier → **`dl_matcher.py`**: a from-scratch neural
  pairwise matcher (DL).
- The hand-tuned probability-threshold decision rule → **`rl_decision.py`**:
  a policy-gradient-trained per-entity stopping policy (RL).

## Why these two specifically

**DL matcher.** Gradient-boosted trees on hand-engineered string-similarity
scores (Levenshtein, Jaccard, TF-IDF cosine, ...) are strong but the
features are fixed in advance — the tree can only recombine them. A neural
matcher instead *learns its own representation* of each name/address
directly from characters, so it can pick up soft, compositional patterns
(e.g. which trigrams matter most for a given script, or how much weight to
give the address versus the name for a given kind of business) that no
single hand-written formula captures. Concretely:

- **Encoder** — character trigrams, hashed into a small learned embedding
  table and mean-pooled (a Deep Averaging Network, Iyyer et al. 2015).
  Hashing + averaging is deliberately simple and typo/reordering-robust,
  and needs no pretrained weights, no tokenizer, no vocabulary file — it
  works identically on Latin, Devanagari, or French text out of the box,
  which matters given the open country/script set.
- **Interaction** — the standard NLI-style `[a, b, |a−b|, a*b]` features
  between the S1 and candidate encodings, for name and address separately.
- **Fusion** — Dense+ReLU, then a learned sigmoid **gate** (Highway
  Networks, Srivastava et al. 2015) that elementwise-scales the hidden
  vector before the output layer, so the network can learn per-pair how
  much to trust the interaction signal.
- **Loss** — binary cross-entropy *plus* a within-S1 pairwise ranking hinge
  term, since only the relative order of candidates within one S1 actually
  determines the accepted set.

It has on the order of 10^6 parameters (dominated by the 32,768×24 hashed
embedding table), no license at all (nothing downloaded — every weight is
trained from zero on the provided data), and satisfies the ≤8B-parameter /
MIT-Apache constraint trivially.

**RL stopping policy.** Macro F0.5 is computed per S1 from a *discrete*
accepted set, with sharp non-smooth special cases (empty-truth-empty-pred
→ 1.0, empty-truth-nonempty-pred → 0.0). It is not a smooth function of any
single candidate's score, so it can't be optimized by gradient descent the
way BCE can — this is exactly the setting policy-gradient RL is designed
for: pick a discrete action sequence to directly maximize a scalar, possibly
non-differentiable reward. We frame "how many sorted candidates to keep"
as an episode: at each step, accept the next candidate or stop; the
terminal reward is the *actual* F0.5 for that entity. The policy is
trained with REINFORCE and a self-critical (batch-mean) baseline for
variance reduction (Rennie et al. 2017), which is a principled alternative
to the closed-form "expected-F0.5" grid search the LightGBM baseline uses.

## Implementation notes / environment constraint

This sandbox has ~1 CPU core, no GPU, and no access to `download.pytorch.org`
or huggingface.co (only package registries are reachable), and a `pip
install torch` here pulls the CUDA build and doesn't fit on the available
disk. Both `dl_matcher.py` and `rl_decision.py` are therefore implemented
in **pure NumPy** with hand-derived forward/backward passes (see
`nn_numpy.py`) — a real, from-scratch trainable neural net and policy
network, just without a deep-learning framework underneath.

`dl_matcher.py` also defines the identical architecture as a
`torch.nn.Module` (`_TorchGatedDualDAN`), used automatically whenever
`torch` is importable (i.e. on the real hackathon machine, which the task
brief says has a GPU) — same design, GPU-batched, real autograd, larger
embedding dims/batches feasible. `DLMatcher` picks whichever backend is
available; nothing else in the pipeline needs to know which one is active.
The torch path is provided as a ready-to-run template and has not been
executed here (no GPU/torch in this sandbox) — verify it in your own
environment before relying on it for the leaderboard.

## Honest results (synthetic smoke-test data, NOT the real dataset)

I don't have the actual multi-million-row training data — it was never
uploaded to this session — so the numbers below are from a small synthetic
dataset (80 train S1 / 79 S2 / 54 S3 rows, ~13% singletons) built only to
verify the pipeline runs correctly end-to-end and that the math is right,
not to estimate real performance:

| Decision rule | Val macro F0.5 |
|---|---|
| DL matcher + fixed threshold (best of a sweep) | 0.292 |
| DL matcher + RL stopping policy | 0.225 |

On this tiny sample the RL policy actually did *worse* than a simple
fixed-threshold sweep on held-out validation, despite its own training
reward trending upward (0.28 → 0.35 over 25 epochs) — a textbook
high-variance REINFORCE symptom on a small entity count (24 val S1 ids).
I'm reporting this rather than hiding it: **per the score-boosters rule
in the task brief ("keep a change only if val improves by ≥0.002"), on
this sample the fixed-threshold rule should be kept, not the RL policy.**
At real dataset scale (millions of pairs, thousands+ of val S1 entities
per batch), REINFORCE variance shrinks substantially and the self-critical
baseline tends to help much more — but that needs to be re-validated on
the real val fold, not assumed. `train_dl_rl.py` prints both numbers every
run specifically so this comparison is never skipped.

Also worth independently sanity-checking on the real data: the BCE loss
curve should look like the smoke-test's (0.69 → falling steadily) — if it
plateaus near 0.69 (=ln 2) on real data, the embedding table or learning
rate needs attention before trusting anything downstream.

## Recommended usage

Treat this as a **candidate third model to ensemble**, not a drop-in
replacement for the validated LightGBM baseline:
1. Run both `train.py` (LightGBM) and `train_dl_rl.py` (DL+RL) on the real
   train fold.
2. Compare val macro F0.5 honestly (`train_dl_rl.py`'s printed
   `baseline_fixed_threshold_f05` vs `rl_policy_f05` vs the LightGBM
   pipeline's own reported score).
3. If either neural variant beats LightGBM by ≥0.002, prefer it; if not,
   its matcher probability is still a cheap extra feature to feed into the
   LightGBM stage-2 model (`dl_score` as one more column in `features.py`),
   which is usually a safer way to capture "at least some of the gain"
   without betting the whole submission on the newer, less-tested path.

## Files added

```
src/
  nn_numpy.py       # generic Dense layer + Adam optimizer, pure NumPy
  dl_matcher.py      # Gated Dual-DAN matcher (numpy backend + torch template)
  rl_decision.py      # REINFORCE stopping policy + rollout/eval
  train_dl_rl.py       # trainer: candidates -> DL matcher -> RL policy -> val F0.5
  predict_dl_rl.py     # test inference -> matching_results.tsv / candidate_pairs.tsv
```

## Commands

```bash
cd src
python3 train_dl_rl.py --train-dir ../dataset/train --model-dir ../model_dl_rl \
    --dl-epochs 10 --rl-epochs 20
python3 predict_dl_rl.py --test-dir ../dataset/test --model-dir ../model_dl_rl \
    --out-dir ../output_dl_rl
# to compare against the fixed-threshold fallback instead of the RL policy:
python3 predict_dl_rl.py --test-dir ../dataset/test --model-dir ../model_dl_rl \
    --out-dir ../output_dl_rl_fixed --no-rl --fixed-threshold 0.5
python3 check_format.py --test-dir ../dataset/test \
    --matching ../output_dl_rl/matching_results.tsv \
    --candidate ../output_dl_rl/candidate_pairs.tsv
```
