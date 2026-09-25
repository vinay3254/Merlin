import numpy as np
import pandas as pd
from src.blocking import build_token_index, token_overlap_candidates, address_prefix_candidates, embedding_knn_candidates, union_candidates, candidates_to_rows


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


def test_token_overlap_candidates_excludes_tokens_over_max_doc_freq():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "acme traders", "business_address": "", "country": "US"}])
    # "acme" appears in 3 other-source rows -- over-cap when max_doc_freq=2
    # "traders" appears in only 1 -- under-cap, should still match
    other = _df([
        {"entity_id": "S2-00001", "business_name": "acme traders", "business_address": "", "country": "US"},
        {"entity_id": "S2-00002", "business_name": "acme corp", "business_address": "", "country": "US"},
        {"entity_id": "S2-00003", "business_name": "acme logistics", "business_address": "", "country": "US"},
    ])
    result = token_overlap_candidates(s1, other, max_doc_freq=2)
    # "acme" is over-cap (freq=3 > 2) and contributes nothing; "traders" (freq=1) still matches S2-00001
    assert result["S1-00001"] == {"S2-00001"}


_EMBED_VOCAB = {}


def _fake_embedder(texts):
    # deterministic stub: identical texts -> identical vectors, no network/model needed
    vectors = []
    for t in texts:
        key = t.strip().lower()
        if key not in _EMBED_VOCAB:
            _EMBED_VOCAB[key] = len(_EMBED_VOCAB)
        idx = _EMBED_VOCAB[key]
        vec = np.zeros(64)
        vec[idx % 64] = 1.0
        vectors.append(vec)
    return np.array(vectors)


def test_embedding_knn_candidates_matches_identical_text():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "acme traders", "business_address": "123 main st", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "acme traders", "business_address": "123 main st", "country": "US"}])
    result = embedding_knn_candidates(s1, s2, embedder=_fake_embedder, top_k=5, min_sim=0.99)
    assert result["S1-00001"] == {"S2-00001"}


def test_embedding_knn_candidates_respects_min_sim_threshold():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "alpha", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "zzz completely different", "business_address": "", "country": "US"}])
    result = embedding_knn_candidates(s1, s2, embedder=_fake_embedder, top_k=5, min_sim=0.99)
    assert result["S1-00001"] == set()


def test_union_candidates_merges_and_dedupes():
    a = {"S1-00001": {"S2-00001", "S2-00002"}}
    b = {"S1-00001": {"S2-00002", "S3-00001"}}
    merged = union_candidates(a, b)
    assert merged["S1-00001"] == {"S2-00001", "S2-00002", "S3-00001"}


def test_union_candidates_handles_key_present_in_only_one_dict():
    a = {"S1-00001": {"S2-00001"}}
    b = {"S1-00002": {"S3-00001"}}
    merged = union_candidates(a, b)
    assert merged["S1-00001"] == {"S2-00001"}
    assert merged["S1-00002"] == {"S3-00001"}


def test_candidates_to_rows_produces_sorted_lists():
    union = {"S1-00001": {"S2-00002", "S2-00001"}}
    rows = candidates_to_rows(union)
    assert rows["S1-00001"] == ["S2-00001", "S2-00002"]
