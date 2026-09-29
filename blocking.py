"""
Scalable blocking (drop-in replacement for the earlier TF-IDF +
NearestNeighbors version).

WHY THE OLD VERSION DIED: it fit a TF-IDF vectorizer and ran brute-force
cosine NearestNeighbors over EVERY record sharing a country label. On a
real dataset with millions of rows per country, that is an O(n^2)
operation in both time and memory (a country block of 1M+2M rows means
building similarity structures over 3M+ documents and searching all of
them) -- this is almost certainly what got SIGKILL'd by the OOM killer.

NEW STRATEGY: partition records into small buckets with cheap composite
keys BEFORE any similarity scoring, so no O(n^2) comparison is ever
attempted across a whole country. Five keys are used (see `_prep`'s
docstring for exactly what each catches): name-prefix, address-digits,
postcode-like, house-number+name-prefix, and a phonetic key on the name's
first token -- mirroring the reference architecture's own finding that
recall jumped from 0.795 to 0.9773 mainly by adding more blocking keys/kNN
passes, not by improving any single one.
shared-key bucket (now small -- a handful to a few hundred records, not
millions), all pairs are scored in ONE vectorized C++ call via
`rapidfuzz.process.cdist(..., workers=-1)` -- no Python-level per-pair
loop, no fitted model held in memory, and no data structure that scales
with the full dataset size rather than the bucket size.

Any bucket that's still too big (a very generic name prefix, or an empty/
common address-digit key) is skipped with a warning rather than allowed to
blow up time or memory -- recall is sacrificed for that slice, but the
alternate key (name vs. digits) gives it a second chance, and this is a
far better trade than crashing the whole run.
"""
import re
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from normalize import (normalize_name, normalize_address, normalize_country,
                        extract_house_number, extract_postcode_like, soundex_lite)

TOP_K = 10
MAX_BUCKET_SIDE = 1500     # skip (with a warning) any bucket bigger than this
SCORE_FLOOR = 55.0         # rapidfuzz token_sort_ratio floor, 0-100 scale
_DIGITS_RE = re.compile(r"\d+")


def _digit_key(addr: str) -> str:
    digits = _DIGITS_RE.findall(addr or "")
    return "".join(sorted(digits)) if digits else ""


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """One-time, vectorized-as-possible normalization + key construction.
    O(n) in time and memory -- no per-row Python object accumulation beyond
    a few new string columns.

    Five keys now, up from two -- mirroring the recall jump the reference
    architecture got from adding more blocking keys/kNN passes (their B0->B1
    table: 0.795 -> 0.9666 recall just from more keys):
      _key_name    : country + first 4 chars of the normalized name
                     (catches exact/near-exact name matches)
      _key_digits  : country + ALL digits in the address, sorted
                     (a broad address-similarity proxy)
      _key_postcode: country + the single longest digit run (>=4 digits)
                     found in the address (a tighter, more standard
                     postcode-like key -- catches cases where _key_digits'
                     "all digits sorted" would differ due to a house
                     number appearing in a different position)
      _key_house_name: country + house number + first 2 chars of name
                     (catches same-building, differently-spelled names)
      _key_phonetic: country + soundex of the name's first token
                     (catches transliteration/typo variants that a
                     prefix-based key like _key_name would miss entirely --
                     directly addresses the Indic-transliteration noise
                     pattern called out in the problem statement)
    """
    df = df.copy()
    df["_name"] = df["business_name"].apply(normalize_name)
    df["_addr"] = df["business_address"].apply(normalize_address)
    df["_country"] = df["country"].apply(normalize_country)
    df.loc[df["_country"] == "", "_country"] = "__unknown__"
    df["_house_no"] = df["business_address"].apply(extract_house_number)
    df["_postcode"] = df["business_address"].apply(extract_postcode_like)
    df["_first_tok"] = df["_name"].str.split(" ").str[0].fillna("")

    df["_key_name"] = (df["_country"] + "|" +
                        df["_name"].str.replace(" ", "", regex=False).str.slice(0, 4))
    df["_key_digits"] = df["_country"] + "|" + df["_addr"].apply(_digit_key)
    df["_key_postcode"] = df["_country"] + "|" + df["_postcode"]
    df["_key_house_name"] = (df["_country"] + "|" + df["_house_no"] + "|" +
                              df["_name"].str.replace(" ", "", regex=False).str.slice(0, 2))
    df["_key_phonetic"] = df["_country"] + "|" + df["_first_tok"].apply(soundex_lite)
    return df


_ALL_KEYS = ("_key_name", "_key_digits", "_key_postcode",
             "_key_house_name", "_key_phonetic")


def _bucket_indices(df: pd.DataFrame, key_col: str) -> dict:
    """key -> np.array of integer positions. Skips empty/near-empty keys
    (e.g. no digits found, or a blank name) since those match almost
    everything and are exactly the buckets we want capped out anyway."""
    g = df.groupby(key_col).indices
    return {k: v for k, v in g.items() if k and not k.endswith("|")}


def _score_bucket(s1_names, s1_ids, other_names, other_ids, source_label,
                   out_rows, oversized_keys, key_repr):
    n1, n2 = len(s1_names), len(other_names)
    if n1 == 0 or n2 == 0:
        return
    if n2 > MAX_BUCKET_SIDE:
        oversized_keys.append((key_repr, n2))
        return
    mat = process.cdist(s1_names, other_names, scorer=fuzz.token_sort_ratio,
                         workers=-1, score_cutoff=SCORE_FLOOR)
    if mat.size == 0:
        return
    k = min(TOP_K, n2)
    top_idx = np.argpartition(-mat, kth=k - 1, axis=1)[:, :k]
    for i in range(n1):
        for j_local in top_idx[i]:
            score = mat[i, j_local]
            if score >= SCORE_FLOOR:
                out_rows.append((s1_ids[i], other_ids[j_local], source_label,
                                  float(score) / 100.0))


def _run_source(s1: pd.DataFrame, other: pd.DataFrame, source_label: str) -> list:
    out_rows, oversized = [], []
    s1_names_arr = s1["_name"].to_numpy()
    s1_ids_arr = s1["entity_id"].to_numpy()
    other_names_arr = other["_name"].to_numpy()
    other_ids_arr = other["entity_id"].to_numpy()

    for key_col in _ALL_KEYS:
        s1_buckets = _bucket_indices(s1, key_col)
        other_buckets = _bucket_indices(other, key_col)
        shared_keys = s1_buckets.keys() & other_buckets.keys()
        for key in shared_keys:
            s1_pos = s1_buckets[key]
            other_pos = other_buckets[key]
            _score_bucket(
                s1_names_arr[s1_pos].tolist(), s1_ids_arr[s1_pos].tolist(),
                other_names_arr[other_pos].tolist(), other_ids_arr[other_pos].tolist(),
                source_label, out_rows, oversized, key)

    if oversized:
        total_skipped = sum(n for _, n in oversized)
        print(f"[blocking] WARNING: skipped {len(oversized)} oversized "
              f"{source_label} bucket(s) (> {MAX_BUCKET_SIDE} candidates on "
              f"the DB side), {total_skipped} DB records affected across "
              f"those buckets. Raise MAX_BUCKET_SIDE if you have RAM/time "
              f"to spare, or tighten the key (e.g. 5-6 chars instead of 4).")
    return out_rows


def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame,
                         source3: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    s1 = _prep(source1)
    s2 = _prep(source2)
    s3 = _prep(source3)
    print(f"[blocking] normalization + key construction done in "
          f"{time.time()-t0:.1f}s")

    rows = []
    t1 = time.time()
    rows.extend(_run_source(s1, s2, "S2"))
    print(f"[blocking] S2 pass done in {time.time()-t1:.1f}s, "
          f"{len(rows)} raw candidate rows so far")
    t2 = time.time()
    rows.extend(_run_source(s1, s3, "S3"))
    print(f"[blocking] S3 pass done in {time.time()-t2:.1f}s, "
          f"{len(rows)} raw candidate rows total")

    if not rows:
        return pd.DataFrame(columns=["source1_entity_id", "other_id", "source",
                                       "name_cosine_sim"])

    df = pd.DataFrame(rows, columns=["source1_entity_id", "other_id", "source",
                                       "name_cosine_sim"])
    # dedupe pairs found by both keys, keeping the higher score -- then cap
    # to TOP_K per (S1, source) -- both done as vectorized sort/groupby ops,
    # NOT a per-group .apply(), which is far too slow at millions of groups
    df = df.sort_values("name_cosine_sim", ascending=False)
    df = df.drop_duplicates(subset=["source1_entity_id", "other_id"], keep="first")
    df = df.sort_values(["source1_entity_id", "source", "name_cosine_sim"],
                         ascending=[True, True, False])
    df["_rank"] = df.groupby(["source1_entity_id", "source"]).cumcount()
    df = df[df["_rank"] < TOP_K].drop(columns="_rank").reset_index(drop=True)
    print(f"[blocking] total time {time.time()-t0:.1f}s, "
          f"{len(df)} final candidate pairs for "
          f"{df['source1_entity_id'].nunique()} S1 entities")
    return df
