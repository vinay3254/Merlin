from collections import defaultdict

import numpy as np
import pandas as pd

from src.normalize import normalize_text, tokenize

STOPWORDS = {
    "the", "and", "of", "inc", "incorporated", "corporation", "corp",
    "limited", "ltd", "company", "co", "private", "pvt", "llc", "llp",
}

MAX_TOKEN_DOC_FREQ = 500
MAX_ADDRESS_PREFIX_DOC_FREQ = 500
TOP_K_PER_STRATEGY = 30
S1_BATCH_SIZE = 50_000


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


def _weighted_top_k_candidates(
    s1_key_table,      # DataFrame with columns ["entity_id_s1", "key"] -- long format, one row per (entity, key) pair
    other_key_table,   # DataFrame with columns ["entity_id_other", "key"], NOT yet capped
    max_doc_freq,
    top_k=TOP_K_PER_STRATEGY,
    batch_size=S1_BATCH_SIZE,
) -> dict:
    """
    Generic per-S1 top-K candidate selection shared by both blocking
    strategies. Weights each key by 1/log(doc_freq_in_other + 2) (reusing
    the doc-frequency already computed for the cap -- not a TF-IDF refit),
    sums matched-key weight per (s1, other) pair, and keeps only the
    top_k highest-weighted candidates per S1 entity. Processes S1 entities
    in batches so peak memory per merge is bounded regardless of total
    row count -- a hot key otherwise still produces (batch s1-rows with
    that key) x (other rows with that key) per batch, so max_doc_freq
    must be applied to `other_key_table` before any merge, not after.
    """
    if s1_key_table.empty or other_key_table.empty:
        return {}

    doc_freq = other_key_table["key"].value_counts()
    allowed_keys = set(doc_freq[doc_freq <= max_doc_freq].index)
    capped_other = other_key_table[other_key_table["key"].isin(allowed_keys)]
    if capped_other.empty:
        return {}

    weights = 1.0 / np.log(doc_freq[doc_freq <= max_doc_freq] + 2)
    weight_map = weights.to_dict()

    s1_ids = s1_key_table["entity_id_s1"].unique()
    result = {}
    for start in range(0, len(s1_ids), batch_size):
        batch_ids = set(s1_ids[start:start + batch_size])
        batch_keys = s1_key_table[s1_key_table["entity_id_s1"].isin(batch_ids)]
        merged = batch_keys.merge(capped_other, on="key")
        if merged.empty:
            continue
        merged["weight"] = merged["key"].map(weight_map)
        scored = merged.groupby(["entity_id_s1", "entity_id_other"])["weight"].sum().reset_index()
        scored = scored.sort_values("weight", ascending=False)
        top = scored.groupby("entity_id_s1", sort=False).head(top_k)
        for s1_id, group in top.groupby("entity_id_s1"):
            result[s1_id] = set(group["entity_id_other"])
    return result


def token_overlap_candidates(
    s1_df,
    other_df,
    max_doc_freq: int = MAX_TOKEN_DOC_FREQ,
    top_k: int = TOP_K_PER_STRATEGY,
    batch_size: int = S1_BATCH_SIZE,
) -> dict:
    s1_tokens = _name_tokens_table(s1_df, id_col_name="entity_id_s1").rename(columns={"token": "key"})
    other_tokens = _name_tokens_table(other_df, id_col_name="entity_id_other").rename(columns={"token": "key"})

    result = {eid: set() for eid in s1_df["entity_id"]}
    result.update(_weighted_top_k_candidates(s1_tokens, other_tokens, max_doc_freq, top_k, batch_size))
    return result


def _address_prefix_key_table(df, id_col_name: str, prefix_len: int) -> pd.DataFrame:
    keys = df["business_address"].map(
        lambda addr: " ".join(tokenize(normalize_text(addr))[:prefix_len]) or None
    )
    table = pd.DataFrame({id_col_name: df["entity_id"].values, "prefix": keys})
    return table.dropna(subset=["prefix"])


def address_prefix_candidates(
    s1_df,
    other_df,
    prefix_len: int = 3,
    max_doc_freq: int = MAX_ADDRESS_PREFIX_DOC_FREQ,
    top_k: int = TOP_K_PER_STRATEGY,
    batch_size: int = S1_BATCH_SIZE,
) -> dict:
    result = {eid: set() for eid in s1_df["entity_id"]}

    s1_keys = _address_prefix_key_table(s1_df, "entity_id_s1", prefix_len).rename(columns={"prefix": "key"})
    other_keys = _address_prefix_key_table(other_df, "entity_id_other", prefix_len).rename(columns={"prefix": "key"})

    result.update(_weighted_top_k_candidates(s1_keys, other_keys, max_doc_freq, top_k, batch_size))
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
