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
