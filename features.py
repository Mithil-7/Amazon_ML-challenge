"""
Pairwise similarity feature engineering (rewritten for scale).

WHY THE OLD VERSION WAS DANGEROUS AT MILLIONS OF ROWS: `build_record_lookup`
built a plain Python dict keyed by entity_id, where every value was itself a
dict containing two Python `set`s of tokens. Python object overhead (dict +
set + string objects) is roughly 1-2 KB PER RECORD once you include the
sets -- for ~12M combined S1+S2+S3 records that is on the order of 10-20GB
just for this one lookup structure, on top of everything else train.py
needs to hold. That would very likely OOM even after fixing blocking.py.

NEW STRATEGY: `build_record_lookup` now returns a small, columnar
pandas DataFrame (entity_id, normalized name, normalized address, country)
instead of a dict of Python objects -- a DataFrame of that shape for 12M
rows is on the order of a few hundred MB, not tens of GB. `compute_features`
joins candidates against these lookup frames with `pd.merge` (a vectorized
hash join, not a Python-level per-row dict access), processed in chunks so
peak memory is bounded by chunk_size regardless of how many candidate pairs
there are in total.

The public function names and call signature are UNCHANGED from before
(`build_record_lookup(df)`, `compute_features(candidates, lookup_s1,
lookup_other)`), so train.py and predict.py need no changes at all.
"""
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import normalize_name, normalize_address, token_set

FEATURE_COLS = [
    "name_cosine_sim",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_len_diff",
    "addr_token_sort_ratio",
    "addr_partial_ratio",
    "addr_jaccard",
    "addr_len_diff",
    "country_match",
]

CHUNK_SIZE = 500_000  # candidate ROWS per chunk -- bounds peak memory


def _jaccard_arrays(names_a, names_b):
    out = np.empty(len(names_a), dtype=np.float32)
    for i, (a, b) in enumerate(zip(names_a, names_b)):
        ta, tb = token_set(a), token_set(b)
        if not ta and not tb:
            out[i] = 1.0
        elif not ta or not tb:
            out[i] = 0.0
        else:
            out[i] = len(ta & tb) / len(ta | tb)
    return out


def build_record_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """entity_id -> normalized fields, as a small columnar DataFrame
    (NOT a dict of Python objects -- see module docstring)."""
    out = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "_name": df["business_name"].apply(normalize_name).to_numpy(),
        "_addr": df["business_address"].apply(normalize_address).to_numpy(),
        "_country": df["country"].astype(str).str.strip().str.lower().to_numpy(),
    })
    return out


def compute_features(candidates: pd.DataFrame, lookup_s1: pd.DataFrame,
                      lookup_other: pd.DataFrame) -> pd.DataFrame:
    """candidates must have ['source1_entity_id', 'other_id',
    'name_cosine_sim'] (and optionally 'label'). Returns candidates with
    FEATURE_COLS added, processed in chunks to bound peak memory."""
    if len(candidates) == 0:
        return candidates.assign(**{c: pd.Series(dtype=np.float32) for c in FEATURE_COLS})

    s1_small = lookup_s1.rename(columns={
        "entity_id": "source1_entity_id", "_name": "_name_s1",
        "_addr": "_addr_s1", "_country": "_country_s1"})
    other_small = lookup_other.rename(columns={
        "entity_id": "other_id", "_name": "_name_o",
        "_addr": "_addr_o", "_country": "_country_o"})
    # dedupe lookups on id in case the same id appears twice across s2+s3
    # concatenation upstream (shouldn't happen, but merge blows up if it does)
    s1_small = s1_small.drop_duplicates(subset="source1_entity_id")
    other_small = other_small.drop_duplicates(subset="other_id")

    n = len(candidates)
    out_chunks = []
    t0 = time.time()
    for start in range(0, n, CHUNK_SIZE):
        chunk = candidates.iloc[start:start + CHUNK_SIZE].merge(
            s1_small, on="source1_entity_id", how="left").merge(
            other_small, on="other_id", how="left")

        for col in ("_name_s1", "_addr_s1", "_name_o", "_addr_o",
                    "_country_s1", "_country_o"):
            chunk[col] = chunk[col].fillna("")

        n1 = chunk["_name_s1"].to_numpy()
        n2 = chunk["_name_o"].to_numpy()
        a1 = chunk["_addr_s1"].to_numpy()
        a2 = chunk["_addr_o"].to_numpy()

        chunk["name_token_sort_ratio"] = [
            fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
        chunk["name_partial_ratio"] = [
            fuzz.partial_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
        chunk["name_jaccard"] = _jaccard_arrays(n1, n2)
        chunk["name_len_diff"] = np.abs(
            np.array([len(x) for x in n1]) - np.array([len(x) for x in n2])
        ).astype(np.float32)

        chunk["addr_token_sort_ratio"] = [
            fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
        chunk["addr_partial_ratio"] = [
            fuzz.partial_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
        chunk["addr_jaccard"] = _jaccard_arrays(a1, a2)
        chunk["addr_len_diff"] = np.abs(
            np.array([len(x) for x in a1]) - np.array([len(x) for x in a2])
        ).astype(np.float32)

        c1 = chunk["_country_s1"].to_numpy()
        c2 = chunk["_country_o"].to_numpy()
        chunk["country_match"] = np.array(
            [(x == y and x != "") for x, y in zip(c1, c2)], dtype=np.float32)

        drop_cols = ["_name_s1", "_addr_s1", "_country_s1",
                     "_name_o", "_addr_o", "_country_o"]
        out_chunks.append(chunk.drop(columns=drop_cols))

        done = min(start + CHUNK_SIZE, n)
        print(f"[features] {done}/{n} candidate rows featurized "
              f"({time.time()-t0:.1f}s elapsed)")

    return pd.concat(out_chunks, ignore_index=True)
