"""
Vectorized pair features (v3).

Records are pre-processed once into per-row string arrays (RecordTable), and
pair features are computed with rapidfuzz.process.cpdist over aligned lists
(C, multi-threaded) and numpy comparisons, instead of a Python loop per
pair. Pairs are passed as integer row positions into the S1 / other tables.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src.normalize_v3 import address_tokens, is_non_latin, name_parts

_POSTAL_RE = re.compile(r"^\d{5,6}$")


_GLUE_WORDS = {"and", "the", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d"}
_FIELDS = ("core", "legal", "nospace", "addr", "house", "postal", "numbers", "last_tok", "nonlatin")


def _process_chunk(args):
    names, addresses = args
    cols = {f: [] for f in _FIELDS}
    for name, address in zip(names, addresses):
        core, legal = name_parts(name)
        atoks = address_tokens(address)
        nums = [t for t in atoks if t.isdigit()]
        cols["core"].append(" ".join(core))
        cols["legal"].append(" ".join(sorted(set(legal) - _GLUE_WORDS)))
        cols["nospace"].append("".join(core))
        cols["addr"].append(" ".join(atoks))
        cols["house"].append(nums[0] if nums else "")
        cols["postal"].append(next((t for t in reversed(nums) if _POSTAL_RE.match(t)), ""))
        cols["numbers"].append(" ".join(sorted(set(nums))))
        cols["last_tok"].append(atoks[-1] if atoks else "")
        cols["nonlatin"].append(is_non_latin(name))
    return cols


class RecordTable:
    """Per-record normalized strings, aligned with the source DataFrame rows."""

    def __init__(self, df: pd.DataFrame, workers: int = 8, chunk: int = 200_000):
        names = df["business_name"].tolist()
        addresses = df["business_address"].tolist()
        jobs = [(names[i:i + chunk], addresses[i:i + chunk]) for i in range(0, len(names), chunk)]
        if workers > 1 and len(jobs) > 1:
            from multiprocessing import get_context
            with get_context("fork").Pool(workers) as pool:
                parts = pool.map(_process_chunk, jobs)
        else:
            parts = [_process_chunk(j) for j in jobs]
        for field in _FIELDS:
            values = [v for part in parts for v in part[field]]
            dtype = bool if field == "nonlatin" else object
            setattr(self, field, np.array(values, dtype=dtype))
        del parts
        self.core_len = np.array([len(c.split()) if c else 0 for c in self.core], dtype=np.int16)
        self.addr_len = np.array([len(a.split()) if a else 0 for a in self.addr], dtype=np.int16)


def _cp(a, b, scorer):
    return process.cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32)


def _both_nonempty(a, b):
    return (a != "") & (b != "")


def pair_features(s1: RecordTable, other: RecordTable, s1_pos: np.ndarray, other_pos: np.ndarray) -> pd.DataFrame:
    ca, cb = s1.core[s1_pos], other.core[other_pos]
    na, nb = s1.nospace[s1_pos], other.nospace[other_pos]
    aa, ab = s1.addr[s1_pos], other.addr[other_pos]
    feats = {
        "name_ratio": _cp(ca, cb, fuzz.ratio),
        "name_token_set": _cp(ca, cb, fuzz.token_set_ratio),
        "name_token_sort": _cp(ca, cb, fuzz.token_sort_ratio),
        "name_partial": _cp(ca, cb, fuzz.partial_ratio),
        "name_jw": _cp(ca, cb, JaroWinkler.normalized_similarity),
        "name_nospace_ratio": _cp(na, nb, fuzz.ratio),
        "name_nospace_partial": _cp(na, nb, fuzz.partial_ratio),
        "legal_ratio": _cp(s1.legal[s1_pos], other.legal[other_pos], fuzz.ratio),
        "addr_ratio": _cp(aa, ab, fuzz.ratio),
        "addr_token_set": _cp(aa, ab, fuzz.token_set_ratio),
        "addr_token_sort": _cp(aa, ab, fuzz.token_sort_ratio),
        "addr_partial": _cp(aa, ab, fuzz.partial_ratio),
        "numbers_token_set": _cp(s1.numbers[s1_pos], other.numbers[other_pos], fuzz.token_set_ratio),
        # Name of one side against the address of the other: catches S2/S3
        # names that are junk but whose address still carries the name.
        "name_a_in_addr_b": _cp(ca, ab, fuzz.partial_ratio),
    }
    ha, hb = s1.house[s1_pos], other.house[other_pos]
    feats["house_match"] = np.where(_both_nonempty(ha, hb), (ha == hb).astype(np.float32), -1.0).astype(np.float32)
    feats["house_ratio"] = np.where(_both_nonempty(ha, hb), _cp(ha, hb, fuzz.ratio), -1.0).astype(np.float32)
    pa, pb = s1.postal[s1_pos], other.postal[other_pos]
    feats["postal_match"] = np.where(_both_nonempty(pa, pb), (pa == pb).astype(np.float32), -1.0).astype(np.float32)
    la, lb = s1.last_tok[s1_pos], other.last_tok[other_pos]
    feats["last_tok_match"] = np.where(_both_nonempty(la, lb), (la == lb).astype(np.float32), -1.0).astype(np.float32)
    feats["name_b_nonlatin"] = other.nonlatin[other_pos].astype(np.float32)
    feats["core_len_a"] = s1.core_len[s1_pos].astype(np.float32)
    feats["core_len_b"] = other.core_len[other_pos].astype(np.float32)
    feats["addr_len_a"] = s1.addr_len[s1_pos].astype(np.float32)
    feats["addr_len_b"] = other.addr_len[other_pos].astype(np.float32)
    feats["other_is_s3"] = np.zeros(len(s1_pos), dtype=np.float32)  # filled by caller
    return pd.DataFrame(feats)


def _rank_gap(keys: np.ndarray, values: np.ndarray):
    frame = pd.DataFrame({"k": keys, "v": values})
    grouped = frame.groupby("k")["v"]
    rank = grouped.rank(ascending=False, method="min").to_numpy(np.float32)
    gap = (grouped.transform("max") - frame["v"]).to_numpy(np.float32)
    size = grouped.transform("size").to_numpy(np.float32)
    return rank, gap, size


def global_blocking_features(s1_pos: np.ndarray, other_pos: np.ndarray, blk_score: np.ndarray) -> pd.DataFrame:
    """
    Context features from the blocking score, computed over the FULL
    candidate set (so "competing S1 entities for this other record" sees
    every S1 entity, not just the ones in the current chunk or training
    subset): rank / gap-to-best within the S1 entity's candidates and within
    the other record's competing S1 entities.
    """
    rank_s, gap_s, _ = _rank_gap(s1_pos, blk_score)
    rank_o, gap_o, n_o = _rank_gap(other_pos, blk_score)
    return pd.DataFrame({
        "blk_score": blk_score.astype(np.float32),
        "blk_rank_in_s1": rank_s, "blk_gap_in_s1": gap_s,
        "blk_rank_in_other": rank_o, "blk_gap_in_other": gap_o, "blk_n_other": n_o,
    })


def chunk_features(s1: RecordTable, other: RecordTable, s1_pos, other_pos, other_is_s3, blk_feats: pd.DataFrame) -> pd.DataFrame:
    """
    String features for a chunk of pairs. The chunk must contain ALL
    candidates of each S1 entity in it (per-S1 context features are computed
    within the chunk). blk_feats: the chunk's rows of global_blocking_features.
    """
    feat = pair_features(s1, other, s1_pos, other_pos)
    feat["other_is_s3"] = np.asarray(other_is_s3, dtype=np.float32)
    for col in blk_feats.columns:
        feat[col] = blk_feats[col].to_numpy()
    feat["name_addr_sum"] = feat["name_token_set"] + feat["addr_token_set"]
    rank, gap, _ = _rank_gap(s1_pos, feat["name_addr_sum"].to_numpy())
    feat["na_rank_in_s1"] = rank
    feat["na_gap_in_s1"] = gap
    return feat
