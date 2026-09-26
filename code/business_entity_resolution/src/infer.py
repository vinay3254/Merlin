import os
import subprocess
import sys

import pandas as pd

from src.blocking import (
    token_overlap_candidates,
    address_token_overlap_candidates,
    embedding_knn_candidates,
    union_candidates,
    candidates_to_rows,
    default_embedder,
)
from src.features import featurize_pairs
from src.train import load_model_artifact
from src.utils_io import read_source_tsv, write_id_mapping_tsv


def generate_candidates(s1_df, s2_df, s3_df, embedder=None, others_df=None) -> dict:
    if others_df is None:
        others_df = pd.concat([s2_df, s3_df], ignore_index=True)
    strategies = [
        token_overlap_candidates(s1_df, others_df),
        address_token_overlap_candidates(s1_df, others_df),
    ]
    if embedder is not None:
        strategies.append(embedding_knn_candidates(s1_df, others_df, embedder=embedder))
    return union_candidates(*strategies)


def score_candidates(s1_df, others_df, candidates: dict, model, feature_columns) -> pd.DataFrame:
    pairs = pd.DataFrame(
        [(s1_id, other_id) for s1_id, other_ids in candidates.items() for other_id in other_ids],
        columns=["source1_entity_id", "other_entity_id"],
    )
    if pairs.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "score"])

    feat_df = featurize_pairs(pairs, s1_df, others_df)
    scores = model.predict_proba(feat_df[feature_columns])[:, 1]
    return pd.DataFrame({
        "source1_entity_id": feat_df["source1_entity_id"].values,
        "other_entity_id": feat_df["other_entity_id"].values,
        "score": scores,
    })


def assign_matches(scored_df: pd.DataFrame, threshold: float) -> dict:
    result = {}
    for s1_id, group in scored_df.groupby("source1_entity_id"):
        matched = group[group["score"] >= threshold]["other_entity_id"].tolist()
        result[s1_id] = matched
    return result


def maybe_run_validator(output_dir: str, test_dir: str, validator_path: str):
    if not os.path.exists(validator_path):
        return None
    result = subprocess.run(
        [
            sys.executable, validator_path,
            "--matching", os.path.join(output_dir, "matching_results.tsv"),
            "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
            "--test-dir", test_dir,
        ],
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def run_inference(dataset_dir: str, model_path: str, output_dir: str, use_embeddings: bool = True) -> None:
    # Fail fast: load the model artifact and source datasets before any
    # expensive blocking/candidate-generation work, so a bad --model-path
    # or missing dataset file errors out immediately.
    artifact = load_model_artifact(model_path)

    s1_df = read_source_tsv(os.path.join(dataset_dir, "test_source1.tsv"))
    s2_df = read_source_tsv(os.path.join(dataset_dir, "test_source2.tsv"))
    s3_df = read_source_tsv(os.path.join(dataset_dir, "test_source3.tsv"))
    others_df = pd.concat([s2_df, s3_df], ignore_index=True)

    embedder = default_embedder if use_embeddings else None
    candidates = generate_candidates(s1_df, s2_df, s3_df, embedder=embedder, others_df=others_df)

    candidate_rows = candidates_to_rows(candidates)
    for entity_id in s1_df["entity_id"]:
        candidate_rows.setdefault(entity_id, [])
    write_id_mapping_tsv(
        candidate_rows, os.path.join(output_dir, "candidate_pairs.tsv"),
        id_col="source1_entity_id", list_col="candidate_entity_ids",
    )

    scored_df = score_candidates(s1_df, others_df, candidates, artifact["model"], artifact["feature_columns"])
    matches = assign_matches(scored_df, artifact["threshold"])

    for entity_id in s1_df["entity_id"]:
        matches.setdefault(entity_id, [])
    write_id_mapping_tsv(
        matches, os.path.join(output_dir, "matching_results.tsv"),
        id_col="source1_entity_id", list_col="matched_entity_ids",
    )

    validator_output = maybe_run_validator(
        output_dir, dataset_dir, os.path.join(os.path.dirname(dataset_dir), "..", "utils", "validate_submission.py")
    )
    if validator_output:
        print(validator_output)
