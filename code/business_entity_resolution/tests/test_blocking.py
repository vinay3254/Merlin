import numpy as np
import pandas as pd
from src.blocking import build_token_index, token_overlap_candidates, address_prefix_candidates, address_token_overlap_candidates, embedding_knn_candidates, union_candidates, candidates_to_rows


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


def test_address_prefix_candidates_excludes_prefixes_over_max_doc_freq():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "", "business_address": "123 MG Road Bangalore", "country": "India"}])
    # "123 mg road" appears in 3 other-source rows -- over-cap when max_doc_freq=2
    other = _df([
        {"entity_id": "S2-00001", "business_name": "", "business_address": "123 MG Road Near SBI ATM", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "", "business_address": "123 MG Road Opposite Mall", "country": "India"},
        {"entity_id": "S2-00003", "business_name": "", "business_address": "123 MG Road Behind Park", "country": "India"},
    ])
    result = address_prefix_candidates(s1, other, prefix_len=3, max_doc_freq=2)
    # over-cap prefix contributes nothing -- no candidates survive
    assert result["S1-00001"] == set()


def test_token_overlap_candidates_keeps_only_top_k_per_s1_entity():
    # S1 entity shares a distinct, low-doc-freq token with each of 4 other-source
    # entities. Each shared token has a different doc_freq (hence weight):
    # rarer tokens (lower doc_freq) get higher weight (1/log(doc_freq+2)).
    s1 = _df([{"entity_id": "S1-00001", "business_name": "alpha bravo charlie delta", "business_address": "", "country": "US"}])
    other = _df([
        # "alpha" doc_freq=1 (highest weight)
        {"entity_id": "S2-00001", "business_name": "alpha", "business_address": "", "country": "US"},
        # "bravo" doc_freq=2
        {"entity_id": "S2-00002", "business_name": "bravo", "business_address": "", "country": "US"},
        {"entity_id": "S2-00003", "business_name": "bravo filler", "business_address": "", "country": "US"},
        # "charlie" doc_freq=3
        {"entity_id": "S2-00004", "business_name": "charlie", "business_address": "", "country": "US"},
        {"entity_id": "S2-00005", "business_name": "charlie filler", "business_address": "", "country": "US"},
        {"entity_id": "S2-00006", "business_name": "charlie filler2", "business_address": "", "country": "US"},
        # "delta" doc_freq=4 (lowest weight)
        {"entity_id": "S2-00007", "business_name": "delta", "business_address": "", "country": "US"},
        {"entity_id": "S2-00008", "business_name": "delta filler", "business_address": "", "country": "US"},
        {"entity_id": "S2-00009", "business_name": "delta filler2", "business_address": "", "country": "US"},
        {"entity_id": "S2-00010", "business_name": "delta filler3", "business_address": "", "country": "US"},
    ])
    result = token_overlap_candidates(s1, other, max_doc_freq=100, top_k=2)
    # Only the highest-weighted 2 candidates should survive: S2-00001 (via "alpha",
    # doc_freq=1) and one of the bravo-matched entities (doc_freq=2, next-highest weight).
    assert len(result["S1-00001"]) == 2
    assert "S2-00001" in result["S1-00001"]


def test_address_prefix_candidates_keeps_only_top_k_per_s1_entity():
    # Address-prefix keys are single-per-entity, so all matches for a given
    # S1 entity share the same prefix (and thus the same weight); this test
    # proves the top_k truncation mechanism (group size capped to top_k)
    # rather than weight-based ordering, which the token-overlap test above
    # already covers via multiple distinct keys.
    s1 = _df([{"entity_id": "S1-00001", "business_name": "", "business_address": "123 MG Road Bangalore", "country": "India"}])
    other = _df([
        {"entity_id": "S2-00001", "business_name": "", "business_address": "123 MG Road Near SBI ATM", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "", "business_address": "123 MG Road Opposite Mall", "country": "India"},
        {"entity_id": "S2-00003", "business_name": "", "business_address": "123 MG Road Behind Park", "country": "India"},
        {"entity_id": "S2-00004", "business_name": "", "business_address": "123 MG Road Beside Bank", "country": "India"},
    ])
    result = address_prefix_candidates(s1, other, prefix_len=3, max_doc_freq=100, top_k=2)
    assert len(result["S1-00001"]) == 2
    assert result["S1-00001"].issubset({"S2-00001", "S2-00002", "S2-00003", "S2-00004"})


def test_token_overlap_candidates_batching_matches_unbatched_result():
    s1 = _df([
        {"entity_id": "S1-00001", "business_name": "Sharma Traders", "business_address": "", "country": "India"},
        {"entity_id": "S1-00002", "business_name": "Totally Unrelated Bakery", "business_address": "", "country": "India"},
    ])
    other = _df([
        {"entity_id": "S2-00001", "business_name": "Sharma Traders Pvt Ltd", "business_address": "", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "Totally Unrelated Bakery Shop", "business_address": "", "country": "India"},
        {"entity_id": "S2-00003", "business_name": "Something Else Entirely", "business_address": "", "country": "India"},
    ])
    unbatched = token_overlap_candidates(s1, other)
    batched = token_overlap_candidates(s1, other, batch_size=1)
    assert batched == unbatched
    assert batched["S1-00001"] == {"S2-00001"}
    assert batched["S1-00002"] == {"S2-00002"}


def test_address_prefix_candidates_batching_matches_unbatched_result():
    s1 = _df([
        {"entity_id": "S1-00001", "business_name": "", "business_address": "123 MG Road Bangalore", "country": "India"},
        {"entity_id": "S1-00002", "business_name": "", "business_address": "77 Park Street Kolkata", "country": "India"},
    ])
    other = _df([
        {"entity_id": "S2-00001", "business_name": "", "business_address": "123 MG Road Near SBI ATM", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "", "business_address": "77 Park Street Near Metro", "country": "India"},
        {"entity_id": "S2-00003", "business_name": "", "business_address": "999 Nowhere Lane", "country": "India"},
    ])
    unbatched = address_prefix_candidates(s1, other, prefix_len=3)
    batched = address_prefix_candidates(s1, other, prefix_len=3, batch_size=1)
    assert batched == unbatched
    assert batched["S1-00001"] == {"S2-00001"}
    assert batched["S1-00002"] == {"S2-00002"}


def test_address_token_overlap_candidates_matches_reordered_abbreviated_address():
    # Real-world pattern: S2/S3 addresses reorder tokens (house number to
    # front) and drop street-level detail -- a prefix match (first N tokens)
    # can't survive this, but full-token overlap still shares "1502" and
    # "faridabad".
    s1 = _df([{"entity_id": "S1-00001", "business_name": "", "business_address": "1502 Tower Olive Omaxe Badhkal Road Sector 43 Faridabad Haryana", "country": "India"}])
    other = _df([{"entity_id": "S2-00001", "business_name": "", "business_address": "B3 1502 Faridabad HR", "country": "India"}])
    result = address_token_overlap_candidates(s1, other)
    assert result["S1-00001"] == {"S2-00001"}


def test_address_token_overlap_candidates_handles_missing_address_without_raising():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    result = address_token_overlap_candidates(s1, s2)
    assert result["S1-00001"] == set()


def test_address_token_overlap_candidates_excludes_tokens_over_max_doc_freq():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "", "business_address": "42 elm street springfield", "country": "US"}])
    # "elm" is over-cap (freq=3 > 2); "springfield" is not shared at all here,
    # "42" doc_freq=1 under-cap -- should still match via "42"
    other = _df([
        {"entity_id": "S2-00001", "business_name": "", "business_address": "42 elm", "country": "US"},
        {"entity_id": "S2-00002", "business_name": "", "business_address": "elm ave", "country": "US"},
        {"entity_id": "S2-00003", "business_name": "", "business_address": "elm court", "country": "US"},
    ])
    result = address_token_overlap_candidates(s1, other, max_doc_freq=2)
    assert result["S1-00001"] == {"S2-00001"}


def test_address_token_overlap_candidates_batching_matches_unbatched_result():
    s1 = _df([
        {"entity_id": "S1-00001", "business_name": "", "business_address": "123 MG Road Bangalore", "country": "India"},
        {"entity_id": "S1-00002", "business_name": "", "business_address": "77 Park Street Kolkata", "country": "India"},
    ])
    other = _df([
        {"entity_id": "S2-00001", "business_name": "", "business_address": "Near SBI ATM 123 MG Road", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "", "business_address": "Near Metro 77 Park Street", "country": "India"},
        {"entity_id": "S2-00003", "business_name": "", "business_address": "999 Nowhere Lane", "country": "India"},
    ])
    unbatched = address_token_overlap_candidates(s1, other)
    batched = address_token_overlap_candidates(s1, other, batch_size=1)
    assert batched == unbatched
    assert "S2-00001" in batched["S1-00001"]
    assert "S2-00002" in batched["S1-00002"]


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
