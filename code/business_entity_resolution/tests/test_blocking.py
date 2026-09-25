import pandas as pd
from src.blocking import build_token_index, token_overlap_candidates, address_prefix_candidates


def _df(rows):
    return pd.DataFrame(rows)


def test_token_overlap_candidates_matches_shared_significant_token():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Sharma Traders", "business_address": "", "country": "India"}])
    s2 = _df([
        {"entity_id": "S2-00001", "business_name": "Sharma Traders Pvt Ltd", "business_address": "", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "Totally Unrelated Bakery", "business_address": "", "country": "India"},
    ])
    result = token_overlap_candidates(s1, s2)
    assert result["S1-00001"] == {"S2-00001"}


def test_token_overlap_candidates_never_includes_source1_ids():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    result = token_overlap_candidates(s1, s2)
    assert all(cid.startswith("S2-") or cid.startswith("S3-") for ids in result.values() for cid in ids)


def test_token_overlap_candidates_no_match_gives_empty_set():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Zephyr Logistics", "business_address": "", "country": "US"}])
    result = token_overlap_candidates(s1, s2)
    assert result["S1-00001"] == set()


def test_address_prefix_candidates_handles_missing_address_without_raising():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    result = address_prefix_candidates(s1, s2)
    assert result["S1-00001"] == set()


def test_address_prefix_candidates_matches_shared_prefix():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "", "business_address": "123 MG Road Bangalore", "country": "India"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "", "business_address": "123 MG Road Near SBI ATM", "country": "India"}])
    result = address_prefix_candidates(s1, s2, prefix_len=3)
    assert result["S1-00001"] == {"S2-00001"}
