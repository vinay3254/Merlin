import re

import pandas as pd
from rapidfuzz import fuzz

from src.normalize import normalize_text, tokenize

_DIGIT_RE = re.compile(r"\d+")


def _safe_text(text) -> str:
    """Convert text to string, handling None and NaN safely."""
    if text is None or (isinstance(text, float) and text != text):  # NaN != NaN
        return ""
    return text


def levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def token_jaccard(a_tokens: list, b_tokens: list) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set, b_set = set(a_tokens), set(b_tokens)
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def common_token_count(a_tokens: list, b_tokens: list) -> int:
    return len(set(a_tokens) & set(b_tokens))


def sorted_tokens_exact_match(a_tokens: list, b_tokens: list) -> int:
    if not a_tokens or not b_tokens:
        return 0
    return int(sorted(a_tokens) == sorted(b_tokens))


def numeric_token_overlap(a_text: str, b_text: str) -> float:
    a_nums = set(_DIGIT_RE.findall(_safe_text(a_text)))
    b_nums = set(_DIGIT_RE.findall(_safe_text(b_text)))
    if not a_nums or not b_nums:
        return 0.0
    union = a_nums | b_nums
    return len(a_nums & b_nums) / len(union)


def name_address_string_features(name_a, addr_a, name_b, addr_b) -> dict:
    norm_name_a, norm_name_b = normalize_text(name_a), normalize_text(name_b)
    norm_addr_a, norm_addr_b = normalize_text(addr_a), normalize_text(addr_b)
    name_tokens_a, name_tokens_b = tokenize(norm_name_a), tokenize(norm_name_b)
    addr_tokens_a, addr_tokens_b = tokenize(norm_addr_a), tokenize(norm_addr_b)

    return {
        "name_levenshtein": levenshtein_ratio(norm_name_a, norm_name_b),
        "name_jaccard": token_jaccard(name_tokens_a, name_tokens_b),
        "name_common_tokens": common_token_count(name_tokens_a, name_tokens_b),
        "name_exact_normalized_match": int(bool(norm_name_a) and norm_name_a == norm_name_b),
        "addr_levenshtein": levenshtein_ratio(norm_addr_a, norm_addr_b),
        "addr_jaccard": token_jaccard(addr_tokens_a, addr_tokens_b),
        "addr_sorted_token_match": sorted_tokens_exact_match(addr_tokens_a, addr_tokens_b),
        "addr_numeric_overlap": numeric_token_overlap(addr_a, addr_b),
    }


def country_match(country_a, country_b) -> int:
    a = _safe_text(country_a).strip().lower()
    b = _safe_text(country_b).strip().lower()
    if not a or not b:
        return 0
    return int(a == b)


def structural_features(name_a, name_b, addr_a, addr_b) -> dict:
    norm_name_a, norm_name_b = normalize_text(name_a), normalize_text(name_b)
    norm_addr_a, norm_addr_b = normalize_text(addr_a), normalize_text(addr_b)
    return {
        "name_length_diff": abs(len(norm_name_a) - len(norm_name_b)),
        "name_token_count_diff": abs(len(tokenize(norm_name_a)) - len(tokenize(norm_name_b))),
        "addr_length_diff": abs(len(norm_addr_a) - len(norm_addr_b)),
        "addr_token_count_diff": abs(len(tokenize(norm_addr_a)) - len(tokenize(norm_addr_b))),
    }


def embedding_cosine_features(vec_a, vec_b) -> dict:
    if vec_a is None or vec_b is None:
        return {"embedding_cosine": 0.0}
    return {"embedding_cosine": float(vec_a @ vec_b)}


def build_pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b, embed_a=None, embed_b=None) -> dict:
    feats = {}
    feats.update(name_address_string_features(name_a, addr_a, name_b, addr_b))
    feats["country_match"] = country_match(country_a, country_b)
    feats.update(structural_features(name_a, name_b, addr_a, addr_b))
    feats.update(embedding_cosine_features(embed_a, embed_b))
    return feats


def featurize_pairs(pairs_df, s1_df, others_df, embed_lookup=None, chunk_size=1_000_000) -> pd.DataFrame:
    """
    Chunked, memory-bounded featurization. Merges pairs_df against s1_df/others_df
    and computes build_pair_features per row, processing chunk_size rows of
    pairs_df at a time so peak memory stays bounded regardless of total pair count.
    Preserves every column already in pairs_df (e.g. "label" if present) plus one
    column per feature key from build_pair_features.
    """
    if pairs_df.empty:
        return pairs_df.copy()

    s1_renamed = s1_df.rename(columns={
        "entity_id": "source1_entity_id", "business_name": "name_a",
        "business_address": "addr_a", "country": "country_a",
    })[["source1_entity_id", "name_a", "addr_a", "country_a"]]
    others_renamed = others_df.rename(columns={
        "entity_id": "other_entity_id", "business_name": "name_b",
        "business_address": "addr_b", "country": "country_b",
    })[["other_entity_id", "name_b", "addr_b", "country_b"]]

    chunk_frames = []
    for start in range(0, len(pairs_df), chunk_size):
        chunk = pairs_df.iloc[start:start + chunk_size]
        merged = chunk.merge(s1_renamed, on="source1_entity_id").merge(others_renamed, on="other_entity_id")

        feature_rows = []
        for row in merged.itertuples():
            embed_a = embed_lookup.get(row.source1_entity_id) if embed_lookup else None
            embed_b = embed_lookup.get(row.other_entity_id) if embed_lookup else None
            feats = build_pair_features(
                row.name_a, row.addr_a, row.country_a,
                row.name_b, row.addr_b, row.country_b,
                embed_a=embed_a, embed_b=embed_b,
            )
            feature_rows.append(feats)

        feat_df = pd.DataFrame(feature_rows)
        for col in pairs_df.columns:
            feat_df[col] = merged[col].values
        chunk_frames.append(feat_df)

    return pd.concat(chunk_frames, ignore_index=True)
