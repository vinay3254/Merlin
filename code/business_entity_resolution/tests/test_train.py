import pandas as pd
from src.train import build_training_pairs, build_feature_matrix, train_model


def _df(rows):
    return pd.DataFrame(rows)


def test_build_training_pairs_includes_ground_truth_positives_even_if_blocking_missed_them():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    others = _df([
        {"entity_id": "S2-00001", "business_name": "Acme Inc", "business_address": "", "country": "US"},
        {"entity_id": "S2-00002", "business_name": "Zephyr", "business_address": "", "country": "US"},
    ])
    gt = _df([{"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001"}])
    blocking_candidates = {"S1-00001": {"S2-00002"}}  # blocking missed the true match, found a false one

    pairs = build_training_pairs(s1, others, gt, blocking_candidates)

    positive = pairs[(pairs["other_entity_id"] == "S2-00001")]
    negative = pairs[(pairs["other_entity_id"] == "S2-00002")]
    assert positive.iloc[0]["label"] == 1
    assert negative.iloc[0]["label"] == 0


def test_build_training_pairs_singleton_produces_no_positive_rows():
    s1 = _df([{"entity_id": "S1-00002", "business_name": "Solo", "business_address": "", "country": "US"}])
    others = _df([{"entity_id": "S2-00003", "business_name": "Unrelated", "business_address": "", "country": "US"}])
    gt = _df([{"source1_entity_id": "S1-00002", "matched_entity_ids": ""}])
    blocking_candidates = {"S1-00002": {"S2-00003"}}

    pairs = build_training_pairs(s1, others, gt, blocking_candidates)
    assert (pairs["label"] == 1).sum() == 0
    assert (pairs["label"] == 0).sum() == 1


def test_build_feature_matrix_produces_one_row_per_pair_with_feature_columns():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "123 Main St", "country": "US"}])
    others = _df([{"entity_id": "S2-00001", "business_name": "Acme Inc", "business_address": "123 Main St", "country": "US"}])
    pairs = _df([{"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "label": 1}])

    matrix = build_feature_matrix(pairs, s1, others)
    assert len(matrix) == 1
    assert "name_levenshtein" in matrix.columns
    assert matrix.iloc[0]["label"] == 1


def test_build_training_pairs_caps_negatives_per_positive():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    others = _df([{"entity_id": f"S2-{i:05d}", "business_name": "Unrelated", "business_address": "", "country": "US"} for i in range(15)])
    gt = _df([{"source1_entity_id": "S1-00001", "matched_entity_ids": ""}])  # no true positives
    blocking_candidates = {"S1-00001": {f"S2-{i:05d}" for i in range(15)}}  # 15 candidates, all negatives

    pairs = build_training_pairs(s1, others, gt, blocking_candidates, max_negatives_per_positive=2, seed=42)

    # cap = 2 * max(1, 0 positives) = 2
    assert (pairs["label"] == 0).sum() == 2
    # sampled negatives must be a subset of the candidate pool
    sampled_ids = set(pairs[pairs["label"] == 0]["other_entity_id"])
    assert sampled_ids.issubset({f"S2-{i:05d}" for i in range(15)})


def test_train_model_fits_without_error_on_small_matrix():
    matrix = _df([
        {"name_levenshtein": 1.0, "addr_levenshtein": 1.0, "country_match": 1, "label": 1},
        {"name_levenshtein": 0.1, "addr_levenshtein": 0.1, "country_match": 0, "label": 0},
        {"name_levenshtein": 0.9, "addr_levenshtein": 0.8, "country_match": 1, "label": 1},
        {"name_levenshtein": 0.05, "addr_levenshtein": 0.0, "country_match": 0, "label": 0},
    ])
    model = train_model(matrix, feature_columns=["name_levenshtein", "addr_levenshtein", "country_match"])
    preds = model.predict_proba(matrix[["name_levenshtein", "addr_levenshtein", "country_match"]])
    assert preds.shape == (4, 2)


import os
import pytest
from src.train import (
    split_train_validation,
    tune_threshold,
    save_model_artifact,
    load_model_artifact,
)


def test_split_train_validation_no_overlap_and_covers_all_ids():
    s1 = _df({"entity_id": [f"S1-{i:05d}" for i in range(10)]})
    train_ids, val_ids = split_train_validation(s1, val_frac=0.2, seed=42)
    assert set(train_ids) & set(val_ids) == set()
    assert set(train_ids) | set(val_ids) == set(s1["entity_id"])
    assert len(val_ids) == 2


def test_split_train_validation_is_deterministic_for_fixed_seed():
    s1 = _df({"entity_id": [f"S1-{i:05d}" for i in range(10)]})
    a = split_train_validation(s1, val_frac=0.2, seed=42)
    b = split_train_validation(s1, val_frac=0.2, seed=42)
    assert a == b


def test_tune_threshold_picks_threshold_that_maximizes_f_beta():
    # true match S2-00001 scores 0.9, false candidate S2-00002 scores 0.4
    scored = _df([
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "score": 0.9},
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00002", "score": 0.4},
    ])
    gt = _df([{"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001"}])
    best = tune_threshold(scored, gt)
    assert 0.4 < best <= 0.9


def test_save_and_load_model_artifact_roundtrip(tmp_path):
    matrix = _df([
        {"f1": 1.0, "label": 1},
        {"f1": 0.0, "label": 0},
        {"f1": 0.9, "label": 1},
        {"f1": 0.1, "label": 0},
    ])
    model = train_model(matrix, feature_columns=["f1"])
    path = str(tmp_path / "model.joblib")
    save_model_artifact(model, threshold=0.55, feature_columns=["f1"], path=path)
    artifact = load_model_artifact(path)
    assert artifact["threshold"] == 0.55
    assert artifact["feature_columns"] == ["f1"]
    assert hasattr(artifact["model"], "predict_proba")


def test_load_model_artifact_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "no_model.joblib"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_model_artifact(str(missing))


def test_run_training_end_to_end_on_small_synthetic_dataset(tmp_path):
    from src.train import run_training

    dataset_dir = tmp_path / "train"
    dataset_dir.mkdir()

    # 5 source-1 entities: 3 have real matches spread across source2/source3,
    # 2 are singletons with no match anywhere. With val_frac=0.2, seed=42 this
    # deterministically splits into train=[S1-00001, S1-00002, S1-00003, S1-00005]
    # and val=[S1-00004], so both the train and validation splits exercise a
    # real positive match as well as a singleton.
    (dataset_dir / "train_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Traders\t123 Main St\tUS\n"
        "S1-00002\tZephyr Corp\t55 Oak Ave\tUS\n"
        "S1-00003\tSolo Business\t1 Lonely Rd\tUS\n"
        "S1-00004\tBright Sun LLC\t22 Sun Blvd\tUS\n"
        "S1-00005\tQuiet Moon Co\t9 Moon St\tUS\n"
    )
    (dataset_dir / "train_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00001\tAcme Traders Inc\t123 Main Street\tUS\n"
        "S2-00002\tBright Sun\t22 Sun Boulevard\tUS\n"
        "S2-00003\tCompletely Unrelated\t999 Nowhere Ave\tUS\n"
    )
    (dataset_dir / "train_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-00001\tZephyr Corporation\t55 Oak Avenue\tUS\n"
        "S3-00002\tAnother Unrelated Place\t1 Nowhere Blvd\tUS\n"
    )
    (dataset_dir / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00001\n"
        "S1-00002\tS3-00001\n"
        "S1-00003\t\n"
        "S1-00004\tS2-00002\n"
        "S1-00005\t\n"
    )

    model_path = tmp_path / "model.joblib"
    summary = run_training(str(dataset_dir), str(model_path), val_frac=0.2, seed=42)

    assert set(summary.keys()) == {"threshold", "n_train_pairs", "n_val_entities", "val_f_beta"}

    artifact = load_model_artifact(str(model_path))
    assert hasattr(artifact["model"], "predict_proba")

    assert isinstance(summary["threshold"], float)
    assert 0.0 <= summary["threshold"] <= 1.0
