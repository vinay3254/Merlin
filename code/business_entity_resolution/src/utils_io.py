import os
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_source_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected source TSV at {path} but it does not exist. "
            "Place the challenge dataset under dataset/train or dataset/test "
            "before running this pipeline."
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df.fillna("")


def read_ground_truth(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected ground truth TSV at {path} but it does not exist."
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in GROUND_TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def parse_id_list(cell) -> list:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == "":
        return []
    return [x for x in str(cell).split(",") if x]


def join_id_list(ids: list) -> str:
    return ",".join(sorted(set(ids)))


def write_id_mapping_tsv(rows: dict, path: str, id_col: str, list_col: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for key in sorted(rows.keys()):
            f.write(f"{key}\t{join_id_list(rows[key])}\n")
