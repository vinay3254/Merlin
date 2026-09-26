"""
Sparse-matrix blocking (v3).

Each record becomes a sparse vector of hashed tokens: core business-name
tokens ("n:" namespace) and address tokens ("a:" namespace), from
src.normalize_v3. Token weight is IDF computed on the other-source side.
Candidates for a batch of S1 rows are the top_k rows of
(S1_batch @ Others_country.T), i.e. the sum of idf^2 over shared tokens.

Why this replaces the pandas merge-based blocking in src/blocking.py:
tokens are int32 column indices in a CSR matrix instead of Python strings
in long DataFrames, so the full 10M-row other side costs ~1-2GB instead of
the 10GB+ that caused repeated thrashing crashes. Tokens with document
frequency above max_df are dropped (they carry little identity and dominate
the join cost), which bounds per-batch work at batch_size * tokens * max_df.
"""
import zlib
from multiprocessing import get_context

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.normalize_v3 import address_tokens, name_parts, name_skeletons

N_FEATURES = 1 << 24
DEFAULT_MAX_DF = 50_000
DEFAULT_TOP_K = 40
DEFAULT_BATCH = 300
NAME_WEIGHT = 1.0
ADDRESS_WEIGHT = 1.0


def _record_token_ids(name, address, fields="all") -> list:
    toks = set()
    if fields in ("all", "name"):
        core, _ = name_parts(name)
        toks.update("n:" + t for t in core)
        if fields == "name":
            # Phonetic consonant skeletons let Indic-script names hit their
            # English originals ("आनंद हेल्थकेयर" / "Anand Healthcare" ->
            # "nd ltkr"). Only in the name-only matrix: added to the combined
            # matrix they dilute it (measured recall 95.5% -> 93.0%).
            toks.update("k:" + t for t in name_skeletons(core))
    if fields in ("all", "addr"):
        toks.update("a:" + t for t in address_tokens(address))
    return [zlib.crc32(t.encode("utf-8")) % N_FEATURES for t in toks]


def _hash_chunk(args):
    names, addresses, fields = args
    indptr = np.zeros(len(names) + 1, dtype=np.int64)
    indices = []
    for i, (name, address) in enumerate(zip(names, addresses)):
        ids = _record_token_ids(name, address, fields)
        indices.extend(ids)
        indptr[i + 1] = indptr[i] + len(ids)
    return indptr, np.asarray(indices, dtype=np.int32)


def hashed_matrix(df: pd.DataFrame, workers: int = 8, chunk: int = 200_000, fields: str = "all") -> sp.csr_matrix:
    """Binary CSR matrix (rows = df rows, cols = hashed tokens). fields: all | name | addr."""
    names = df["business_name"].tolist()
    addresses = df["business_address"].tolist()
    jobs = [(names[i:i + chunk], addresses[i:i + chunk], fields) for i in range(0, len(names), chunk)]
    if workers > 1 and len(jobs) > 1:
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_hash_chunk, jobs)
    else:
        parts = [_hash_chunk(j) for j in jobs]
    indptr_parts, index_parts, offset = [np.zeros(1, dtype=np.int64)], [], 0
    for indptr, indices in parts:
        indptr_parts.append(indptr[1:] + offset)
        index_parts.append(indices)
        offset += len(indices)
    indptr = np.concatenate(indptr_parts)
    indices = np.concatenate(index_parts) if index_parts else np.zeros(0, dtype=np.int32)
    data = np.ones(len(indices), dtype=np.float32)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(len(names), N_FEATURES))
    matrix.sum_duplicates()  # hash collisions inside one record
    matrix.data[:] = 1.0
    return matrix


def _l2_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags((1.0 / norms).astype(np.float32)) @ matrix


_WORKER_STATE = {}


def _search_batch(rows):
    """Top-k search for one batch of S1 row positions (runs in a forked worker)."""
    st = _WORKER_STATE
    q = st["s1_matrix"][rows].multiply(st["idf"]).tocsr()
    q.eliminate_zeros()
    if st["normalize"]:
        q = _l2_normalize(q)
    product = (q @ st["o_t"]).tocsr()
    top_k = st["top_k"]
    out_s1, out_other, out_score = [], [], []
    for local_row in range(product.shape[0]):
        start, end = product.indptr[local_row], product.indptr[local_row + 1]
        if start == end:
            continue
        data = product.data[start:end]
        cols = product.indices[start:end]
        if end - start > top_k:
            keep = np.argpartition(-data, top_k)[:top_k]
            data, cols = data[keep], cols[keep]
        out_s1.append(np.full(len(cols), rows[local_row], dtype=np.int32))
        out_other.append(st["other_rows"][cols].astype(np.int32))
        out_score.append(data.astype(np.float32))
    if not out_s1:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, np.zeros(0, dtype=np.float32)
    return np.concatenate(out_s1), np.concatenate(out_other), np.concatenate(out_score)


def sparse_candidate_pairs(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    max_df: int = DEFAULT_MAX_DF,
    batch_size: int = DEFAULT_BATCH,
    workers: int = 8,
    s1_matrix=None,
    other_matrix=None,
    normalize: bool = True,
    log=None,
):
    """
    Returns (s1_pos, other_pos, score) int32/int32/float32 arrays: row
    positions into s1_df / other_df plus the blocking similarity, up to
    top_k per S1 row, searched within the same country (S1 rows whose country
    has no other-side rows search every country). Positions instead of id
    strings keep ~90M candidate pairs at ~1GB rather than ~6GB of Python
    strings. Batches run in forked workers that share the (read-only)
    other-side matrix copy-on-write.
    """
    if s1_matrix is None:
        s1_matrix = hashed_matrix(s1_df, workers=workers)
    if other_matrix is None:
        other_matrix = hashed_matrix(other_df, workers=workers)

    s1_country = s1_df["country"].to_numpy()
    other_country = other_df["country"].to_numpy()
    parts_s1, parts_other, parts_score = [], [], []

    for country in pd.unique(s1_country):
        s1_rows = np.flatnonzero(s1_country == country)
        other_rows = np.flatnonzero(other_country == country)
        if len(other_rows) == 0:
            other_rows = np.arange(len(other_df))
        o_mat = other_matrix[other_rows]
        doc_freq = np.bincount(o_mat.indices, minlength=N_FEATURES)
        idf = np.log((len(other_rows) + 1) / (doc_freq + 1)).astype(np.float32)
        idf[doc_freq > max_df] = 0.0
        o_mat = o_mat.multiply(idf).tocsr()
        o_mat.eliminate_zeros()
        if normalize:
            o_mat = _l2_normalize(o_mat)
        _WORKER_STATE.update(
            s1_matrix=s1_matrix, idf=idf, o_t=o_mat.T.tocsr(), other_rows=other_rows,
            top_k=top_k, normalize=normalize,
        )
        del o_mat
        batches = [s1_rows[i:i + batch_size] for i in range(0, len(s1_rows), batch_size)]
        if workers > 1 and len(batches) > 1:
            with get_context("fork").Pool(workers) as pool:
                for i, (a, b, c) in enumerate(pool.imap(_search_batch, batches, chunksize=4)):
                    parts_s1.append(a); parts_other.append(b); parts_score.append(c)
                    if log and i % 500 == 0:
                        log(f"blocking {country}: batch {i}/{len(batches)}")
        else:
            for rows in batches:
                a, b, c = _search_batch(rows)
                parts_s1.append(a); parts_other.append(b); parts_score.append(c)
        _WORKER_STATE.clear()

    if not parts_s1:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, np.zeros(0, dtype=np.float32)
    return np.concatenate(parts_s1), np.concatenate(parts_other), np.concatenate(parts_score)


def sparse_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, **kwargs) -> dict:
    """Dict form of sparse_candidate_pairs: {s1_entity_id: set(other_entity_id)}."""
    s1_pos, other_pos, _ = sparse_candidate_pairs(s1_df, other_df, **kwargs)
    s1_ids = s1_df["entity_id"].to_numpy()
    other_ids = other_df["entity_id"].to_numpy()
    result = {eid: set() for eid in s1_ids}
    for s, o in zip(s1_ids[s1_pos], other_ids[other_pos]):
        result[s].add(o)
    return result


STRATEGIES = (("all", 40), ("name", 20), ("addr", 20))


def multi_strategy_pairs(s1_df, other_df, strategies=STRATEGIES, max_df=DEFAULT_MAX_DF, workers=8, log=None):
    """
    Union of per-strategy top-k lists (combined name+address, name-only with
    phonetic skeletons, address-only). Returns (s1_pos, other_pos, scores)
    where scores is a float32 array of shape (n_pairs, n_strategies), 0 where
    a strategy did not retrieve the pair. Measured on a 20k-entity train
    sample: 96.45% recall of true pairs vs 95.5% for the combined list alone.
    """
    keys_parts, strat_parts, score_parts = [], [], []
    n_other = len(other_df)
    for idx, (fields, top_k) in enumerate(strategies):
        om = hashed_matrix(other_df, workers=workers, fields=fields)
        sm = hashed_matrix(s1_df, workers=workers, fields=fields)
        if log:
            log(f"strategy {fields} top_k={top_k}: matrices built")
        a, b, sc = sparse_candidate_pairs(
            s1_df, other_df, top_k=top_k, max_df=max_df, workers=workers,
            s1_matrix=sm, other_matrix=om, log=log,
        )
        del om, sm
        keys_parts.append(a.astype(np.int64) * n_other + b)
        strat_parts.append(np.full(len(a), idx, dtype=np.int8))
        score_parts.append(sc)
    keys = np.concatenate(keys_parts); del keys_parts
    strat = np.concatenate(strat_parts); del strat_parts
    score = np.concatenate(score_parts); del score_parts
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    del keys
    scores = np.zeros((len(unique_keys), len(strategies)), dtype=np.float32)
    scores[inverse, strat] = score
    s1_pos = (unique_keys // n_other).astype(np.int32)
    other_pos = (unique_keys % n_other).astype(np.int32)
    return s1_pos, other_pos, scores
