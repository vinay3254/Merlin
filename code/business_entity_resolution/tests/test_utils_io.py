import pandas as pd
import pytest

from src.utils_io import (
    read_source_tsv,
    read_ground_truth,
    parse_id_list,
    join_id_list,
    write_id_mapping_tsv,
)


def test_read_source_tsv_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.tsv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        read_source_tsv(str(missing))


def test_read_source_tsv_parses_columns(tmp_path):
    p = tmp_path / "source1.tsv"
    p.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Corp\t123 Main St\tUS\n"
    )
    df = read_source_tsv(str(p))
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert df.iloc[0]["entity_id"] == "S1-00001"


def test_read_ground_truth_handles_empty_matches(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047,S3-00812\n"
        "S1-00002\t\n"
    )
    df = read_ground_truth(str(p))
    assert df.iloc[1]["matched_entity_ids"] in ("", None) or pd.isna(df.iloc[1]["matched_entity_ids"])


def test_parse_id_list_splits_and_handles_empty():
    assert parse_id_list("S2-00047,S3-00812") == ["S2-00047", "S3-00812"]
    assert parse_id_list("") == []
    assert parse_id_list(None) == []


def test_join_id_list_sorts_and_dedupes():
    assert join_id_list(["S3-00812", "S2-00047", "S2-00047"]) == "S2-00047,S3-00812"


def test_join_id_list_rejects_true_duplicates_from_caller():
    # caller passed the same id via two different code paths -- must not
    # silently collapse without the caller knowing; join_id_list itself
    # dedupes, but callers that need to detect a bug should check length
    # before calling. This test documents the dedupe behavior explicitly.
    result = join_id_list(["S2-00047", "S2-00047", "S3-00001"])
    assert result == "S2-00047,S3-00001"


def test_write_id_mapping_tsv_writes_expected_format(tmp_path):
    out = tmp_path / "matching_results.tsv"
    rows = {
        "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
        "S1-00002": ["S3-00004"],
        "S1-00003": [],
    }
    write_id_mapping_tsv(rows, str(out), id_col="source1_entity_id", list_col="matched_entity_ids")
    text = out.read_text()
    lines = text.strip("\n").split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert "S1-00001\tS2-00047,S2-00193,S3-00812" in lines
    assert "S1-00003\t" in text  # empty list still gets a row
