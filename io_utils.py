"""IO helpers: loading source/ground-truth tsvs and writing submission tsvs."""
import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.fillna("")


def load_ground_truth(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    return df


def ground_truth_to_pairs(gt: pd.DataFrame) -> set:
    """Return a set of (source1_entity_id, matched_id) positive pairs."""
    pairs = set()
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        ids = row["matched_entity_ids"]
        if not ids:
            continue
        for mid in ids.split(","):
            mid = mid.strip()
            if mid:
                pairs.add((s1, mid))
    return pairs


def write_id_list_tsv(df_pairs: pd.DataFrame, s1_ids, out_path,
                       id_col_name: str):
    """
    df_pairs: DataFrame with columns ['source1_entity_id', 'other_id']
    s1_ids: iterable of ALL source1 entity ids that must appear (incl. those
            with zero candidates/matches -> empty string row).
    """
    grouped = (
        df_pairs.groupby("source1_entity_id")["other_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # dedup, keep order
        .to_dict()
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{grouped.get(s1, '')}\n")
