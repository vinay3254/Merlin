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

from src.blocking_v3 import hashed_matrix, sparse_candidate_pairs
from src.features_v3 import RecordTable, chunk_features, global_blocking_features
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
    return z["s1_pos"].astype(np.int64), z["other_pos"].astype(np.int64), z["score"]


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
    return chunk_features(
        s1_table, other_table, s1_pos[rows], other_pos[rows], other_pos[rows] >= n_s2,
        blk_global.iloc[rows].reset_index(drop=True),
    )


def rows_for_entities(s1_pos, entity_positions):
    return np.flatnonzero(np.isin(s1_pos, entity_positions))


def cmd_train(args):
    import lightgbm as lgb

    s1, others, n_s2 = load_split(args.dataset_dir, "train")
    s1_pos, other_pos, blk = load_candidates(args.candidates)
    log(f"candidates {len(s1_pos)}")
    true_keys, truth = ground_truth_keys(args.dataset_dir, s1, others)
    labels = np.isin(s1_pos * len(others) + other_pos, true_keys)
    log(f"candidate recall over all true pairs: {labels.sum() / len(true_keys):.4f}")
    blk_global = global_blocking_features(s1_pos, other_pos, blk)

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
    x_tr = features_for_rows(tr_rows, s1_pos, other_pos, blk_global, s1_table, other_table, n_s2)
    y_tr = labels[tr_rows]
    x_va = features_for_rows(va_rows, s1_pos, other_pos, blk_global, s1_table, other_table, n_s2)
    y_va = labels[va_rows]
    feature_columns = list(x_tr.columns)
    log(f"train pairs {len(x_tr)} (pos {y_tr.sum()}), val pairs {len(x_va)} (pos {y_va.sum()})")

    model = lgb.LGBMClassifier(
        n_estimators=args.rounds, learning_rate=0.05, num_leaves=255, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=args.workers, verbose=-1,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.log_evaluation(100)])
    log("model trained")

    scored = pd.DataFrame({
        "source1_entity_id": s1_ids[s1_pos[va_rows]],
        "other_entity_id": other_ids[other_pos[va_rows]],
        "score": model.predict_proba(x_va[feature_columns])[:, 1],
    })
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


def cmd_infer(args):
    art = joblib.load(args.model_path)
    threshold = args.threshold if args.threshold is not None else art["threshold"]
    s1, others, n_s2 = load_split(args.dataset_dir, "test")
    s1_pos, other_pos, blk = load_candidates(args.candidates)
    blk_global = global_blocking_features(s1_pos, other_pos, blk)
    s1_ids = s1["entity_id"].to_numpy()
    other_ids = others["entity_id"].to_numpy()
    s1_table = RecordTable(s1, workers=args.workers)
    other_table = RecordTable(others, workers=args.workers)
    del others
    gc.collect()
    log("record tables built")

    # Chunk by S1 entity so every chunk holds all candidates of its entities.
    order = np.argsort(s1_pos, kind="stable")
    s1_pos, other_pos = s1_pos[order], other_pos[order]
    blk_global = blk_global.iloc[order].reset_index(drop=True)
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
    candidates = scored.groupby("source1_entity_id", observed=True)["other_entity_id"].agg(list).to_dict()
    for eid in s1_ids:
        candidates.setdefault(eid, [])
    write_id_mapping_tsv(candidates, os.path.join(args.output_dir, "candidate_pairs.tsv"),
                         id_col="source1_entity_id", list_col="candidate_entity_ids")
    matches = assign_matches(scored, threshold)
    for eid in s1_ids:
        matches.setdefault(eid, [])
    write_id_mapping_tsv(matches, os.path.join(args.output_dir, "matching_results.tsv"),
                         id_col="source1_entity_id", list_col="matched_entity_ids")
    log(f"wrote outputs to {args.output_dir} at threshold {threshold}")


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
    t = sub.add_parser("train")
    t.add_argument("--dataset-dir", required=True)
    t.add_argument("--candidates", required=True)
    t.add_argument("--model-path", required=True)
    t.add_argument("--train-entities", type=int, default=400_000)
    t.add_argument("--val-entities", type=int, default=60_000)
    t.add_argument("--rounds", type=int, default=600)
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
    args = parser.parse_args(argv)
    {"block": cmd_block, "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
