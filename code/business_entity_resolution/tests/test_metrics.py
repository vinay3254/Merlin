import math
from src.metrics import f_beta_score, macro_f_beta


def test_f_beta_score_matches_spec_example():
    predicted = {"S2-00047", "S2-00193", "S3-00812"}
    actual = {"S2-00047", "S3-00812"}
    score = f_beta_score(predicted, actual, beta=0.5)
    assert math.isclose(score, 0.714, abs_tol=0.001)


def test_f_beta_score_singleton_correct_empty_prediction_is_one():
    assert f_beta_score(set(), set(), beta=0.5) == 1.0


def test_f_beta_score_singleton_false_merge_is_zero():
    assert f_beta_score({"S2-00001"}, set(), beta=0.5) == 0.0


def test_f_beta_score_missed_all_matches_is_zero():
    assert f_beta_score(set(), {"S2-00001"}, beta=0.5) == 0.0


def test_f_beta_score_perfect_match_is_one():
    assert f_beta_score({"S2-00001", "S3-00002"}, {"S2-00001", "S3-00002"}, beta=0.5) == 1.0


def test_macro_f_beta_averages_across_entities_including_singletons():
    predictions = {
        "S1-00001": {"S2-00047", "S3-00812"},
        "S1-00002": set(),
    }
    actuals = {
        "S1-00001": {"S2-00047", "S3-00812"},
        "S1-00002": set(),
    }
    assert macro_f_beta(predictions, actuals) == 1.0


def test_macro_f_beta_uses_empty_set_for_missing_prediction_key():
    predictions = {}
    actuals = {"S1-00001": {"S2-00047"}}
    assert macro_f_beta(predictions, actuals) == 0.0
