import numpy as np
import pandas as pd

from src.blocking_v3 import sparse_candidates
from src.features_v3 import RecordTable, chunk_features, global_blocking_features
from src.normalize_v3 import address_tokens, is_non_latin, name_parts


def _df(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_name_parts_splits_legal_forms_including_dotted_and_french():
    assert name_parts("Gagny (France) Club (S.A.R.L.)") == (["gagny", "france", "club"], ["sarl"])
    assert name_parts("Acme Traders Pvt Ltd") == (["acme", "traders"], ["pvt", "ltd"])


def test_name_parts_keeps_indic_words_whole():
    core, legal = name_parts("सनराइज इंडस्ट्रीज लिमिटेड")
    assert core == ["सनराइज", "इंडस्ट्रीज"]
    assert legal == ["लिमिटेड"]


def test_address_tokens_strip_leading_zeros_and_map_states_and_abbreviations():
    assert address_tokens("03415 Lewis Ln, Amarillo, Texas") == ["3415", "lewis", "lane", "amarillo", "tx"]
    assert address_tokens("N° 50 R. DE LA BENAUGE, BORDEAUX") == ["50", "rue", "benauge", "bordeaux"]
    assert address_tokens("1 Silverlake Blvd, New York") == ["1", "silverlake", "boulevard", "ny"]


def test_is_non_latin():
    assert is_non_latin("सनराइज")
    assert not is_non_latin("Crème Brûlée")


def test_sparse_candidates_finds_reordered_address_match_within_country():
    s1 = _df([
        ["S1-1", "Coastal Aluminum", "1435 Timberlake Road, Lexington, TN", "US"],
        ["S1-2", "Maison Hubert SARL", "2 Rue du Blanc Seau, Tourcoing", "France"],
    ])
    others = _df([
        ["S2-1", "Coasta1 Aluminum", "TN, 001435 Timberlake Rd, LEXINGTON", "US"],
        ["S3-1", "MAISON HUBERT", "TOURCOING, 2 R. DU BLANC SEAU", "France"],
        ["S3-2", "Maison Hubert", "2 Rue du Blanc Seau, Tourcoing", "US"],
        # Filler so each country partition has >1 row (idf is 0 for a token
        # present in every row of its partition).
        ["S3-3", "Boulangerie Petit", "8 Avenue Foch, Lyon", "France"],
        ["S3-4", "Garage Central", "40 Quai Perrache, Lyon", "France"],
    ])
    result = sparse_candidates(s1, others, top_k=5, max_df=100, workers=1)
    assert result["S1-1"] == {"S2-1"}
    assert result["S1-2"] == {"S3-1"}


def test_chunk_features_match_across_formatting_noise():
    a = RecordTable(_df([["S1-1", "Sharon Regional Association", "13 Prince Way, Sharon, MA", "US"]]), workers=1)
    b = RecordTable(_df([
        ["S2-1", "SHARONREGIONALASSOCIATION.COM", "SHARON, 13 Prince Way, MA", "US"],
        ["S2-2", "Unrelated Bakery", "99 Other Road, Boston, MA", "US"],
    ]), workers=1)
    s1_pos, other_pos = np.array([0, 0]), np.array([0, 1])
    blk = global_blocking_features(s1_pos, other_pos, np.array([0.9, 0.2], dtype=np.float32))
    feat = chunk_features(a, b, s1_pos, other_pos, np.zeros(2), blk)
    assert feat.loc[0, "name_nospace_partial"] == 100
    assert feat.loc[0, "addr_token_set"] == 100
    assert feat.loc[0, "house_match"] == 1
    assert feat.loc[1, "house_match"] == 0
    assert feat.loc[0, "na_rank_in_s1"] == 1
    assert feat.loc[1, "blk_rank_in_s1"] == 2
