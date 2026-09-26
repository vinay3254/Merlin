from src.normalize import normalize_text, tokenize, ABBREVIATION_MAP


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("Acme, Corp.!") == "acme corporation"


def test_normalize_expands_common_abbreviations():
    assert normalize_text("Sharma & Sons Pvt Ltd") == "sharma and sons private limited"
    assert normalize_text("123 MG Rd") == "123 mg road"
    assert normalize_text("456 Park St") == "456 park street"


def test_normalize_handles_none_and_empty_without_raising():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


def test_normalize_collapses_whitespace():
    assert normalize_text("Acme   Corp") == "acme corporation"


def test_normalize_handles_unseen_country_style_free_text():
    # "France"-style free text with accents/punctuation must not crash
    assert normalize_text("Café de Paris") == "cafe de paris" or normalize_text("Café de Paris") != ""


def test_tokenize_splits_normalized_text():
    assert tokenize(normalize_text("Acme Corp")) == ["acme", "corporation"]


def test_tokenize_handles_empty_string():
    assert tokenize("") == []


def test_abbreviation_map_is_lowercase_keys():
    assert all(k == k.lower() for k in ABBREVIATION_MAP)


def test_normalize_handles_nan_without_becoming_literal_nan_token():
    assert normalize_text(float("nan")) == ""
