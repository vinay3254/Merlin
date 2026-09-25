import pandas as pd

from src.normalize import normalize_text, tokenize

STOPWORDS = {
    "the", "and", "of", "inc", "incorporated", "corporation", "corp",
    "limited", "ltd", "company", "co", "private", "pvt", "llc", "llp",
}

MAX_TOKEN_DOC_FREQ = 5000


def _name_tokens_table(df, id_col_name: str, text_col: str = "business_name") -> pd.DataFrame:
    # .map (one Python call per row) instead of iterrows (no per-row Series
    # construction); explode() is a vectorized pandas op, not a Python loop.
    token_lists = df[text_col].map(
        lambda t: [tok for tok in tokenize(normalize_text(t)) if tok not in STOPWORDS]
    )
    table = pd.DataFrame({id_col_name: df["entity_id"].values, "token": token_lists})
    return table.explode("token").dropna(subset=["token"])


def build_token_index(df, id_col="entity_id", text_col="business_name") -> dict:
    table = _name_tokens_table(df, id_col_name="entity_id", text_col=text_col)
    index = {}
    for token, group in table.groupby("token")["entity_id"]:
        index[token] = set(group)
    return index


def token_overlap_candidates(s1_df, other_df, max_doc_freq: int = MAX_TOKEN_DOC_FREQ) -> dict:
    s1_tokens = _name_tokens_table(s1_df, id_col_name="entity_id_s1")
    other_tokens = _name_tokens_table(other_df, id_col_name="entity_id_other")

    doc_freq = other_tokens["token"].value_counts()
    allowed_tokens = set(doc_freq[doc_freq <= max_doc_freq].index)
    other_tokens = other_tokens[other_tokens["token"].isin(allowed_tokens)]

    result = {eid: set() for eid in s1_df["entity_id"]}
    if s1_tokens.empty or other_tokens.empty:
        return result

    merged = s1_tokens.merge(other_tokens, on="token")
    if merged.empty:
        return result

    grouped = merged.groupby("entity_id_s1")["entity_id_other"].agg(set)
    result.update(grouped.to_dict())
    return result


def _address_prefix_key_table(df, id_col_name: str, prefix_len: int) -> pd.DataFrame:
    keys = df["business_address"].map(
        lambda addr: " ".join(tokenize(normalize_text(addr))[:prefix_len]) or None
    )
    table = pd.DataFrame({id_col_name: df["entity_id"].values, "prefix": keys})
    return table.dropna(subset=["prefix"])


def address_prefix_candidates(s1_df, other_df, prefix_len: int = 3) -> dict:
    result = {eid: set() for eid in s1_df["entity_id"]}

    s1_keys = _address_prefix_key_table(s1_df, "entity_id_s1", prefix_len)
    other_keys = _address_prefix_key_table(other_df, "entity_id_other", prefix_len)
    if s1_keys.empty or other_keys.empty:
        return result

    merged = s1_keys.merge(other_keys, on="prefix")
    if merged.empty:
        return result

    grouped = merged.groupby("entity_id_s1")["entity_id_other"].agg(set)
    result.update(grouped.to_dict())
    return result
