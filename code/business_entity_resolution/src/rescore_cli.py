"""
Re-score an existing candidate_pairs.tsv with a trained model artifact and
write every (source1_entity_id, other_entity_id, score) row to a scores TSV.

This skips blocking entirely (the ~2hr GPU/memory-heavy part of inference),
so assignment/threshold experiments can be re-run from the saved scores in
minutes. The model's features do not use embeddings (embedding_cosine is in
EXCLUDED_FEATURE_COLUMNS), so scores match what run_inference would produce
for the same candidates.

Memory: the parent holds the source TSVs once; candidate pairs are streamed
in chunks of --chunk-entities S1 entities, joined against the source text in
the parent, and only the joined chunk is shipped to a worker process.
"""
import argparse
import os
import sys
import time
from itertools import islice
from multiprocessing import get_context

import pandas as pd

from src.features import build_pair_features
from src.train import load_model_artifact
from src.utils_io import read_source_tsv

_MODEL = None
_FEATURE_COLUMNS = None


def _init_worker(model_path):
    global _MODEL, _FEATURE_COLUMNS
    artifact = load_model_artifact(model_path)
    _MODEL = artifact["model"]
    _FEATURE_COLUMNS = artifact["feature_columns"]


def _score_chunk(merged: pd.DataFrame) -> pd.DataFrame:
    columns = None
    for row in merged.itertuples(index=False):
        feats = build_pair_features(
            row.name_a, row.addr_a, row.country_a,
            row.name_b, row.addr_b, row.country_b,
        )
        if columns is None:
            columns = {key: [] for key in feats}
        for key, value in feats.items():
            columns[key].append(value)
    feat_df = pd.DataFrame(columns)
    scores = _MODEL.predict_proba(feat_df[_FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame({
        "source1_entity_id": merged["source1_entity_id"].values,
        "other_entity_id": merged["other_entity_id"].values,
        "score": scores.astype("float32"),
    })


def _iter_joined_chunks(candidates_path, s1_text, others_text, chunk_entities):
    reader = pd.read_csv(
        candidates_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_entities,
    )
    for chunk in reader:
        pairs = chunk.assign(
            other_entity_id=chunk["candidate_entity_ids"].str.split(",")
        ).explode("other_entity_id")
        pairs = pairs[pairs["other_entity_id"] != ""][["source1_entity_id", "other_entity_id"]]
        merged = pairs.merge(s1_text, on="source1_entity_id").merge(others_text, on="other_entity_id")
        yield merged


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-entities", type=int, default=20_000)
    args = parser.parse_args(argv)

    s1_df = read_source_tsv(os.path.join(args.dataset_dir, "test_source1.tsv"))
    others_df = pd.concat([
        read_source_tsv(os.path.join(args.dataset_dir, "test_source2.tsv")),
        read_source_tsv(os.path.join(args.dataset_dir, "test_source3.tsv")),
    ], ignore_index=True)
    s1_text = s1_df.rename(columns={
        "entity_id": "source1_entity_id", "business_name": "name_a",
        "business_address": "addr_a", "country": "country_a",
    })[["source1_entity_id", "name_a", "addr_a", "country_a"]]
    others_text = others_df.rename(columns={
        "entity_id": "other_entity_id", "business_name": "name_b",
        "business_address": "addr_b", "country": "country_b",
    })[["other_entity_id", "name_b", "addr_b", "country_b"]]
    del s1_df, others_df

    start = time.time()
    n_rows = 0
    first = True
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.model_path,)) as pool, \
            open(args.scores_out, "w") as out:
        chunks = _iter_joined_chunks(args.candidates, s1_text, others_text, args.chunk_entities)
        # Pool.imap would drain the whole generator into its task queue up
        # front (all ~38M joined pairs in memory at once), so feed it one
        # bounded wave of `workers` chunks at a time instead.
        while True:
            wave = list(islice(chunks, args.workers))
            if not wave:
                break
            for scored in pool.map(_score_chunk, wave):
                scored.to_csv(out, sep="\t", index=False, header=first, float_format="%.6f")
                first = False
                n_rows += len(scored)
            del wave
            elapsed = time.time() - start
            print(f"{n_rows} pairs scored, {elapsed:.0f}s, {n_rows / elapsed:.0f} pairs/s", flush=True)
    print(f"Wrote {args.scores_out}")


if __name__ == "__main__":
    sys.exit(main())
