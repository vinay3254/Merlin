"""
v3 pipeline: sparse blocking -> vectorized features -> LightGBM ->
one-S1-per-record assignment.

  python -m src.pipeline_v3 block --split train --dataset-dir DIR --out cands_train.npz
  python -m src.pipeline_v3 train --dataset-dir DIR --candidates cands_train.npz --model-path model_v3.joblib
  python -m src.pipeline_v3 block --split test  --dataset-dir DIR --out cands_test.npz
  python -m src.pipeline_v3 infer --dataset-dir DIR --candidates cands_test.npz --model-path model_v3.joblib --output-dir output_v3

Candidates are stored as row positions into (source1, concat(source2,
source3)) as read by read_source_tsv, which is deterministic for a given
dataset directory.
"""
import argparse
import gc
import os
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd

from src.blocking_v3 import hashed_matrix, multi_strategy_pairs, sparse_candidate_pairs
from src.features_v3 import (
    RecordTable, chunk_features, global_blocking_features, global_context_features, iter_global_blocking,
    iter_global_context,
)
from src.infer import assign_matches
from src.metrics import macro_f_beta
from src.train import split_train_validation
from src.utils_io import read_source_tsv, write_id_mapping_tsv

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def load_split(dataset_dir, split):
    s1 = read_source_tsv(os.path.join(dataset_dir, f"{split}_source1.tsv"))
    s2 = read_source_tsv(os.path.join(dataset_dir, f"{split}_source2.tsv"))
    s3 = read_source_tsv(os.path.join(dataset_dir, f"{split}_source3.tsv"))
    n_s2 = len(s2)
    others = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    return s1, others, n_s2


def cmd_block(args):
    s1, others, _ = load_split(args.dataset_dir, args.split)
    log(f"loaded s1={len(s1)} others={len(others)}")
    if args.multi:
        s1_pos, other_pos, scores = multi_strategy_pairs(s1, others, max_df=args.max_df, workers=args.workers, log=log)
        np.savez(args.out, s1_pos=s1_pos, other_pos=other_pos, score=scores[:, 0], extra_scores=scores[:, 1:])
    else:
        om = hashed_matrix(others, workers=args.workers)
        sm = hashed_matrix(s1, workers=args.workers)
        log(f"hashed others nnz={om.nnz}")
        s1_pos, other_pos, score = sparse_candidate_pairs(
            s1, others, top_k=args.top_k, max_df=args.max_df, workers=args.workers,
            s1_matrix=sm, other_matrix=om, log=log,
        )
        np.savez(args.out, s1_pos=s1_pos, other_pos=other_pos, score=score)
    log(f"saved {len(s1_pos)} pairs to {args.out}")


def load_candidates(path):
    z = np.load(path)
    extra = z["extra_scores"] if "extra_scores" in z.files else None
    # int32 positions: ~110M pairs x 2 arrays is 0.9GB instead of 1.8GB.
    return z["s1_pos"].astype(np.int32), z["other_pos"].astype(np.int32), z["score"], extra


def extra_score_features(extra, rows=None):
    """Name-only / address-only blocking scores (0 = not retrieved by that list)."""
    if extra is None:
        return pd.DataFrame()
    sel = extra if rows is None else extra[rows]
    return pd.DataFrame({
        "blk_name_score": sel[:, 0], "blk_addr_score": sel[:, 1],
        "blk_n_lists": (sel > 0).sum(axis=1).astype(np.float32),
    })


def ground_truth_keys(dataset_dir, s1, others):
    """Set of true pairs as int64 keys s1_pos * n_others + other_pos, and per-S1 truth sets."""
    gt = pd.read_csv(os.path.join(dataset_dir, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    truth = {sid: set(m.split(",")) - {""} for sid, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    pairs = gt.assign(o=gt["matched_entity_ids"].str.split(",")).explode("o")
    pairs = pairs[pairs["o"] != ""]
    s1_index = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    other_index = pd.Series(np.arange(len(others)), index=others["entity_id"].to_numpy())
    sp = s1_index.reindex(pairs["source1_entity_id"].to_numpy()).to_numpy()
    op = other_index.reindex(pairs["o"].to_numpy()).to_numpy()
    ok = ~(np.isnan(sp) | np.isnan(op))
    keys = sp[ok].astype(np.int64) * len(others) + op[ok].astype(np.int64)
    return np.unique(keys), truth


def features_for_rows(rows, s1_pos, other_pos, blk_global, s1_table, other_table, n_s2):
    if isinstance(blk_global, GlobalRows):
        ctx = blk_global.iloc_rows(rows)
    else:
        ctx = blk_global.iloc[rows].reset_index(drop=True)
    return chunk_features(
        s1_table, other_table, s1_pos[rows], other_pos[rows], other_pos[rows] >= n_s2, ctx,
    )


def rows_for_entities(s1_pos, entity_positions):
    return np.flatnonzero(np.isin(s1_pos, entity_positions))


def cmd_train(args):
    import lightgbm as lgb

    s1, others, n_s2 = load_split(args.dataset_dir, "train")
    s1_pos, other_pos, blk, extra = load_candidates(args.candidates)
    log(f"candidates {len(s1_pos)}")
    true_keys, truth = ground_truth_keys(args.dataset_dir, s1, others)
    labels = np.isin(s1_pos.astype(np.int64) * len(others) + other_pos, true_keys)
    log(f"candidate recall over all true pairs: {labels.sum() / len(true_keys):.4f}")

    train_ids, val_ids = split_train_validation(s1)
    rng = random.Random(args.seed)
    pos_of = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    train_ents = pos_of[rng.sample(train_ids, min(args.train_entities, len(train_ids)))].to_numpy()
    val_ents = pos_of[rng.sample(val_ids, min(args.val_entities, len(val_ids)))].to_numpy()

    s1_ids = s1["entity_id"].to_numpy()
    other_ids = others["entity_id"].to_numpy()
    s1_table = RecordTable(s1, workers=args.workers)
    other_table = RecordTable(others, workers=args.workers)
    del others
    gc.collect()
    log("record tables built")

    tr_rows = rows_for_entities(s1_pos, train_ents)
    va_rows = rows_for_entities(s1_pos, val_ents)
    keep = np.concatenate([tr_rows, va_rows])
    ctx = {name: values[keep] for name, values in iter_global_blocking(s1_pos, other_pos, blk)}
    ctx.update({name: values[keep] for name, values in iter_global_context(s1_table, other_table, s1_pos, other_pos)})
    ctx = pd.concat([pd.DataFrame(ctx), extra_score_features(extra, keep)], axis=1)
    gc.collect()
    log("global context features built")
    n_tr = len(tr_rows)
    x_tr = chunk_features(s1_table, other_table, s1_pos[tr_rows], other_pos[tr_rows],
                          other_pos[tr_rows] >= n_s2, ctx.iloc[:n_tr].reset_index(drop=True))
    y_tr = labels[tr_rows]
    x_va = chunk_features(s1_table, other_table, s1_pos[va_rows], other_pos[va_rows],
                          other_pos[va_rows] >= n_s2, ctx.iloc[n_tr:].reset_index(drop=True))
    y_va = labels[va_rows]
    del ctx
    feature_columns = list(x_tr.columns)
    log(f"train pairs {len(x_tr)} (pos {y_tr.sum()}), val pairs {len(x_va)} (pos {y_va.sum()})")

    model = lgb.LGBMClassifier(
        n_estimators=args.rounds, learning_rate=0.05, num_leaves=255, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=args.workers, verbose=-1,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)],
              callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)])
    log("model trained")

    scored = pd.DataFrame({
        "source1_entity_id": s1_ids[s1_pos[va_rows]],
        "other_entity_id": other_ids[other_pos[va_rows]],
        "score": model.predict_proba(x_va[feature_columns])[:, 1],
    })
    np.savez(args.model_path + ".val.npz", s1_pos=s1_pos[va_rows], other_pos=other_pos[va_rows],
             score=scored["score"].to_numpy(np.float32), label=y_va, val_ents=val_ents)
    val_truth = {s1_ids[p]: truth.get(s1_ids[p], set()) for p in val_ents}
    best = (0.0, 0.5)
    for threshold in np.arange(0.2, 0.9, 0.05):
        matches = assign_matches(scored, threshold)
        f = macro_f_beta({k: set(v) for k, v in matches.items()}, val_truth)
        log(f"val threshold={threshold:.2f} F0.5={f:.4f}")
        best = max(best, (f, float(threshold)))
    log(f"best val F0.5={best[0]:.4f} at threshold={best[1]:.2f}")
    importances = sorted(zip(model.feature_importances_, feature_columns), reverse=True)
    log("feature importance: " + ", ".join(f"{c}={i}" for i, c in importances))
    joblib.dump({"model": model, "threshold": best[1], "feature_columns": feature_columns}, args.model_path)
    log(f"saved {args.model_path}")


def memmap_global_features(s1_table, other_table, s1_pos, other_pos, blk, extra, path):
    """
    Writes every global (full-candidate-set) feature column into an on-disk
    float32 memmap of shape (n_pairs, n_columns) one column at a time, so
    ~110M test pairs x ~20 columns (~9GB) never have to sit in RAM at once.
    Returns (memmap, column_names).
    """
    def columns():
        yield from iter_global_blocking(s1_pos, other_pos, blk)
        yield from iter_global_context(s1_table, other_table, s1_pos, other_pos)
        extra_df = extra_score_features(extra)
        for col in extra_df.columns:
            yield col, extra_df[col].to_numpy()

    names = list(global_blocking_features(s1_pos[:1], other_pos[:1], blk[:1]).columns)
    names += [n for n, _ in iter_global_context_names()]
    names += list(extra_score_features(extra[:1] if extra is not None else None).columns)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mm = np.lib.format.open_memmap(path + ".npy", mode="w+", dtype=np.float32, shape=(len(s1_pos), len(names)))
    for col, values in columns():
        mm[:, names.index(col)] = values
        del values
    mm.flush()
    return mm, names


def iter_global_context_names():
    for prefix in ("gname", "gaddr"):
        for suffix in ("other", "s1"):
            yield f"{prefix}_rank_in_{suffix}", None
            yield f"{prefix}_gap_in_{suffix}", None
    yield "gname_n_near_identical_in_other", None
    yield "gname_best", None


class GlobalRows:
    """Row access into the global-feature memmap through a row permutation."""

    def __init__(self, mm_and_names, order):
        self.mm, self.names = mm_and_names
        self.order = order

    def iloc_rows(self, rows):
        idx = self.order[rows]
        sort = np.argsort(idx)
        block = np.empty((len(idx), len(self.names)), dtype=np.float32)
        block[sort] = self.mm[idx[sort]]
        return pd.DataFrame(block, columns=self.names)


def cmd_infer(args):
    art = joblib.load(args.model_path)
    threshold = args.threshold if args.threshold is not None else art["threshold"]
    s1, others, n_s2 = load_split(args.dataset_dir, "test")
    s1_pos, other_pos, blk, extra = load_candidates(args.candidates)
    s1_ids = s1["entity_id"].to_numpy()
    other_ids = others["entity_id"].to_numpy()
    s1_table = RecordTable(s1, workers=args.workers)
    other_table = RecordTable(others, workers=args.workers)
    del others
    gc.collect()
    log("record tables built")
    blk_global = memmap_global_features(
        s1_table, other_table, s1_pos, other_pos, blk, extra,
        os.path.join(args.output_dir, "global_features.f32"),
    )
    del extra
    gc.collect()
    log("global context features built")

    # Chunk by S1 entity so every chunk holds all candidates of its entities.
    order = np.argsort(s1_pos, kind="stable")
    s1_pos, other_pos = s1_pos[order], other_pos[order]
    blk_global = GlobalRows(blk_global, order)
    boundaries = np.flatnonzero(np.diff(s1_pos)) + 1
    starts = np.concatenate([[0], boundaries])
    scores = np.zeros(len(s1_pos), dtype=np.float32)
    step = args.chunk_pairs
    begin = 0
    while begin < len(s1_pos):
        end = min(begin + step, len(s1_pos))
        if end < len(s1_pos):
            end = starts[np.searchsorted(starts, end, side="right") - 1] or len(s1_pos)
        rows = np.arange(begin, end)
        feat = features_for_rows(rows, s1_pos, other_pos, blk_global, s1_table, other_table, n_s2)
        scores[begin:end] = art["model"].predict_proba(feat[art["feature_columns"]])[:, 1]
        log(f"scored {end}/{len(s1_pos)}")
        begin = end

    scored = pd.DataFrame({
        "source1_entity_id": pd.Categorical(s1_ids[s1_pos]),
        "other_entity_id": pd.Categorical(other_ids[other_pos]),
        "score": scores,
    })
    os.makedirs(args.output_dir, exist_ok=True)
    scored.to_csv(os.path.join(args.output_dir, "scores.tsv.gz"), sep="\t", index=False, float_format="%.5f")
    write_outputs(scored, s1_ids, threshold, args.output_dir)


def write_outputs(scored, s1_ids, threshold, output_dir):
    """candidate_pairs.tsv (every scored pair) + matching_results.tsv (assigned)."""
    candidates = {eid: [] for eid in s1_ids}
    for s1_id, other_id in zip(scored["source1_entity_id"].astype(str), scored["other_entity_id"].astype(str)):
        candidates[s1_id].append(other_id)
    write_id_mapping_tsv(candidates, os.path.join(output_dir, "candidate_pairs.tsv"),
                         id_col="source1_entity_id", list_col="candidate_entity_ids")
    del candidates
    matches = assign_matches(scored, threshold)
    matches = {str(k): [str(v) for v in vs] for k, vs in matches.items()}
    for eid in s1_ids:
        matches.setdefault(eid, [])
    write_id_mapping_tsv(matches, os.path.join(output_dir, "matching_results.tsv"),
                         id_col="source1_entity_id", list_col="matched_entity_ids")
    log(f"wrote outputs to {output_dir} at threshold {threshold}")


def cmd_finalize(args):
    """Rebuild outputs from a saved scores.tsv.gz (e.g. at another threshold)."""
    scored = pd.read_csv(
        os.path.join(args.output_dir, "scores.tsv.gz"), sep="\t", keep_default_na=False,
        dtype={"source1_entity_id": "category", "other_entity_id": "category", "score": "float32"},
    )
    s1_ids = read_source_tsv(os.path.join(args.dataset_dir, "test_source1.tsv"))["entity_id"].to_numpy()
    write_outputs(scored, s1_ids, args.threshold, args.output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("block")
    b.add_argument("--split", choices=["train", "test"], required=True)
    b.add_argument("--dataset-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--top-k", type=int, default=40)
    b.add_argument("--max-df", type=int, default=50_000)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--multi", action="store_true", help="union of combined / name-only / address-only lists")
    t = sub.add_parser("train")
    t.add_argument("--dataset-dir", required=True)
    t.add_argument("--candidates", required=True)
    t.add_argument("--model-path", required=True)
    t.add_argument("--train-entities", type=int, default=200_000)
    t.add_argument("--val-entities", type=int, default=60_000)
    t.add_argument("--rounds", type=int, default=1500)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--workers", type=int, default=10)
    i = sub.add_parser("infer")
    i.add_argument("--dataset-dir", required=True)
    i.add_argument("--candidates", required=True)
    i.add_argument("--model-path", required=True)
    i.add_argument("--output-dir", required=True)
    i.add_argument("--threshold", type=float, default=None)
    i.add_argument("--chunk-pairs", type=int, default=4_000_000)
    i.add_argument("--workers", type=int, default=10)
    f = sub.add_parser("finalize")
    f.add_argument("--dataset-dir", required=True)
    f.add_argument("--output-dir", required=True)
    f.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args(argv)
    {"block": cmd_block, "train": cmd_train, "infer": cmd_infer, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
