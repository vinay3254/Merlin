from collections import defaultdict

import numpy as np
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


FLAT_INDEX_THRESHOLD = 1000  # below this many rows, exact search is cheap enough


def default_embedder(texts: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = getattr(default_embedder, "_model", None)
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        default_embedder._model = model
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors)


def _combined_texts(df) -> list:
    names = df["business_name"].map(normalize_text)
    addrs = df["business_address"].map(normalize_text)
    return [f"{n} {a}".strip() for n, a in zip(names, addrs)]


def _search_partition(s1_df, other_df, embedder, top_k: int, min_sim: float) -> dict:
    import faiss

    s1_ids = s1_df["entity_id"].tolist()
    other_ids = other_df["entity_id"].tolist()

    s1_vecs = np.ascontiguousarray(embedder(_combined_texts(s1_df)), dtype="float32")
    other_vecs = np.ascontiguousarray(embedder(_combined_texts(other_df)), dtype="float32")
    dim = other_vecs.shape[1]

    if len(other_ids) < FLAT_INDEX_THRESHOLD:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = max(1, min(int(len(other_ids) ** 0.5), 4096))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(other_vecs)
        index.nprobe = min(nlist, 10)
    index.add(other_vecs)

    k = min(top_k, len(other_ids))
    sims, indices = index.search(s1_vecs, k)

    result = {}
    for i, s1_id in enumerate(s1_ids):
        candidates = {
            other_ids[idx]
            for idx, sim in zip(indices[i], sims[i])
            if idx != -1 and sim >= min_sim
        }
        result[s1_id] = candidates
    return result


def embedding_knn_candidates(s1_df, other_df, embedder=default_embedder, top_k: int = 10, min_sim: float = 0.5) -> dict:
    result = {eid: set() for eid in s1_df["entity_id"]}
    if len(s1_df) == 0 or len(other_df) == 0:
        return result

    s1_country = s1_df["country"].fillna("")
    other_country = other_df["country"].fillna("")
    for country, s1_group in s1_df.groupby(s1_country):
        other_group = other_df[other_country == country]
        if len(other_group) == 0:
            continue
        result.update(_search_partition(s1_group, other_group, embedder, top_k, min_sim))
    return result


def union_candidates(*candidate_dicts) -> dict:
    merged = defaultdict(set)
    for d in candidate_dicts:
        for key, ids in d.items():
            merged[key] |= ids
    return dict(merged)


def candidates_to_rows(union_dict: dict) -> dict:
    return {key: sorted(ids) for key, ids in union_dict.items()}
