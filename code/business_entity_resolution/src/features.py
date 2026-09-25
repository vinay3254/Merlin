import re

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
