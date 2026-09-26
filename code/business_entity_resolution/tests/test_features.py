from src.features import (
    levenshtein_ratio,
    token_jaccard,
    common_token_count,
    sorted_tokens_exact_match,
    numeric_token_overlap,
    name_address_string_features,
)


def test_levenshtein_ratio_identical_strings_is_one():
    assert levenshtein_ratio("acme corp", "acme corp") == 1.0


def test_levenshtein_ratio_empty_string_is_zero():
    assert levenshtein_ratio("", "acme") == 0.0
    assert levenshtein_ratio("acme", "") == 0.0


def test_token_jaccard_partial_overlap():
    assert token_jaccard(["acme", "traders"], ["acme", "logistics"]) == 1 / 3


def test_token_jaccard_empty_list_is_zero():
    assert token_jaccard([], ["acme"]) == 0.0
    assert token_jaccard([], []) == 0.0


def test_common_token_count():
    assert common_token_count(["a", "b", "c"], ["b", "c", "d"]) == 2


def test_sorted_tokens_exact_match_handles_reordering():
    assert sorted_tokens_exact_match(["main", "st", "123"], ["123", "main", "st"]) == 1
    assert sorted_tokens_exact_match(["main", "st"], ["oak", "ave"]) == 0
    assert sorted_tokens_exact_match([], []) == 0


def test_numeric_token_overlap_handles_missing_numbers_without_raising():
    assert numeric_token_overlap("Near SBI ATM", "Landmark reference only") == 0.0
    assert numeric_token_overlap("", "") == 0.0


def test_numeric_token_overlap_matches_shared_numbers():
    assert numeric_token_overlap("123 Main St 560001", "123 Main Road") > 0.0


def test_name_address_string_features_returns_all_keys():
    feats = name_address_string_features("Acme Corp", "123 Main St", "Acme Corporation", "123 Main Street")
    expected_keys = {
        "name_levenshtein", "name_jaccard", "name_common_tokens",
        "name_exact_normalized_match", "addr_levenshtein", "addr_jaccard",
        "addr_sorted_token_match", "addr_numeric_overlap",
    }
    assert set(feats.keys()) == expected_keys
    assert feats["name_exact_normalized_match"] == 1  # both normalize to "acme corporation"


def test_numeric_token_overlap_handles_nan_without_raising():
    assert numeric_token_overlap(float("nan"), "123 Main St") == 0.0
    assert numeric_token_overlap(float("nan"), float("nan")) == 0.0


import numpy as np
from src.features import (
    country_match,
    structural_features,
    embedding_cosine_features,
    build_pair_features,
)


def test_country_match_exact_and_case_insensitive():
    assert country_match("US", "us") == 1
    assert country_match("India", "US") == 0


def test_country_match_handles_unseen_country_value_like_france():
    assert country_match("France", "France") == 1
    assert country_match("France", "US") == 0


def test_country_match_empty_is_zero():
    assert country_match("", "US") == 0
    assert country_match("", "") == 0


def test_country_match_handles_nan_without_raising():
    assert country_match(float("nan"), "US") == 0
    assert country_match("US", float("nan")) == 0
    assert country_match(float("nan"), float("nan")) == 0


def test_structural_features_computes_diffs():
    from src.normalize import normalize_text
    feats = structural_features("Acme Corp", "Acme", "123 Main St", "123 Main")
    # Normalized: "Acme Corp" -> "acme corporation", "Acme" -> "acme"
    norm_a = normalize_text("Acme Corp")
    norm_b = normalize_text("Acme")
    assert feats["name_length_diff"] == abs(len(norm_a) - len(norm_b))
    assert feats["name_token_count_diff"] >= 0


def test_embedding_cosine_features_handles_none_vectors():
    assert embedding_cosine_features(None, None)["embedding_cosine"] == 0.0


def test_embedding_cosine_features_computes_dot_product():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert embedding_cosine_features(a, b)["embedding_cosine"] == 1.0


def test_build_pair_features_merges_all_feature_groups():
    feats = build_pair_features(
        "Acme Corp", "123 Main St", "US",
        "Acme Corporation", "123 Main Street", "US",
        embed_a=np.array([1.0, 0.0]), embed_b=np.array([1.0, 0.0]),
    )
    assert "name_levenshtein" in feats
    assert "country_match" in feats and feats["country_match"] == 1
    assert "name_length_diff" in feats
    assert "embedding_cosine" in feats and feats["embedding_cosine"] == 1.0


import pandas as pd
from src.features import featurize_pairs


def test_featurize_pairs_chunking_matches_unchunked_output_with_no_lost_or_duplicated_rows():
    s1 = pd.DataFrame([
        {"entity_id": f"S1-{i:05d}", "business_name": f"Acme {i}", "business_address": f"{i} Main St", "country": "US"}
        for i in range(5)
    ])
    others = pd.DataFrame([
        {"entity_id": f"S2-{i:05d}", "business_name": f"Acme {i} Inc", "business_address": f"{i} Main Street", "country": "US"}
        for i in range(5)
    ])
    pairs = pd.DataFrame([
        {"source1_entity_id": f"S1-{i:05d}", "other_entity_id": f"S2-{i:05d}", "label": i % 2}
        for i in range(5)
    ])

    chunked = featurize_pairs(pairs, s1, others, chunk_size=2)
    unchunked = featurize_pairs(pairs, s1, others, chunk_size=1000)

    assert len(chunked) == 5
    assert len(chunked) == len(unchunked)

    chunked_sorted = chunked.sort_values("source1_entity_id").reset_index(drop=True)
    unchunked_sorted = unchunked.sort_values("source1_entity_id").reset_index(drop=True)

    assert list(chunked_sorted["source1_entity_id"]) == list(pairs["source1_entity_id"])
    assert list(chunked_sorted["other_entity_id"]) == list(pairs["other_entity_id"])
    assert list(chunked_sorted["label"]) == list(pairs["label"])
    pd.testing.assert_frame_equal(chunked_sorted, unchunked_sorted)
