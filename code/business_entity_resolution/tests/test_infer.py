import pandas as pd
import pytest

from src.infer import generate_candidates, score_candidates, assign_matches, run_inference
from src.train import train_model


def _df(rows):
    return pd.DataFrame(rows)


def test_generate_candidates_unions_token_and_address_strategies_without_embedder():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme Traders", "business_address": "123 Main St", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme Traders Inc", "business_address": "999 Other Rd", "country": "US"}])
    s3 = _df([{"entity_id": "S3-00001", "business_name": "Nothing Alike", "business_address": "123 Main Street", "country": "US"}])

    result = generate_candidates(s1, s2, s3, embedder=None)
    assert result["S1-00001"] == {"S2-00001", "S3-00001"}


def test_generate_candidates_never_returns_source1_ids():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s3 = _df([{"entity_id": "S3-00001", "business_name": "Acme", "business_address": "", "country": "US"}])

    result = generate_candidates(s1, s2, s3, embedder=None)
    assert all(cid.startswith(("S2-", "S3-")) for cid in result["S1-00001"])


def test_assign_matches_gives_empty_list_for_entity_with_no_scores_above_threshold():
    scored = _df([{"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "score": 0.2}])
    result = assign_matches(scored, threshold=0.5)
    assert result["S1-00001"] == []


def test_assign_matches_keeps_scores_above_threshold():
    scored = _df([
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "score": 0.9},
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00002", "score": 0.1},
    ])
    result = assign_matches(scored, threshold=0.5)
    assert result["S1-00001"] == ["S2-00001"]


def test_run_inference_writes_row_for_every_source1_entity_including_singletons(tmp_path):
    dataset_dir = tmp_path / "test"
    dataset_dir.mkdir()
    (dataset_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Traders\t123 Main St\tUS\n"
        "S1-00002\tCompletely Unique Business\t999 Nowhere Ave\tFrance\n"
    )
    (dataset_dir / "test_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00001\tAcme Traders Inc\t123 Main Street\tUS\n"
    )
    (dataset_dir / "test_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    )

    matrix = _df([
        {"name_levenshtein": 1.0, "label": 1},
        {"name_levenshtein": 0.0, "label": 0},
    ])
    model = train_model(matrix, feature_columns=["name_levenshtein"])
    model_path = tmp_path / "model.joblib"
    from src.train import save_model_artifact
    save_model_artifact(model, threshold=0.5, feature_columns=["name_levenshtein"], path=str(model_path))

    output_dir = tmp_path / "output"
    run_inference(str(dataset_dir), str(model_path), str(output_dir), use_embeddings=False)

    results_text = (output_dir / "matching_results.tsv").read_text()
    assert "S1-00001" in results_text
    assert "S1-00002\t" in results_text  # singleton still gets a row

    candidates_text = (output_dir / "candidate_pairs.tsv").read_text()
    assert "S1-00001" in candidates_text


def test_run_inference_missing_dataset_file_fails_fast(tmp_path):
    dataset_dir = tmp_path / "test"
    dataset_dir.mkdir()
    output_dir = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        run_inference(str(dataset_dir), str(tmp_path / "model.joblib"), str(output_dir), use_embeddings=False)
