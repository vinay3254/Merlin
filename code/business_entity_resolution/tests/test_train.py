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
