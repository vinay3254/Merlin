import os

import pandas as pd

from src.blocking import (
    token_overlap_candidates,
    address_prefix_candidates,
    embedding_knn_candidates,
    union_candidates,
    candidates_to_rows,
    default_embedder,
)
from src.features import build_pair_features
from src.train import load_model_artifact
from src.utils_io import read_source_tsv, write_id_mapping_tsv


def generate_candidates(s1_df, s2_df, s3_df, embedder=None) -> dict:
    others_df = pd.concat([s2_df, s3_df], ignore_index=True)
    strategies = [
        token_overlap_candidates(s1_df, others_df),
        address_prefix_candidates(s1_df, others_df),
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

    s1_renamed = s1_df.rename(columns={
        "entity_id": "source1_entity_id", "business_name": "name_a",
        "business_address": "addr_a", "country": "country_a",
    })[["source1_entity_id", "name_a", "addr_a", "country_a"]]
    others_renamed = others_df.rename(columns={
        "entity_id": "other_entity_id", "business_name": "name_b",
        "business_address": "addr_b", "country": "country_b",
    })[["other_entity_id", "name_b", "addr_b", "country_b"]]

    merged = pairs.merge(s1_renamed, on="source1_entity_id").merge(others_renamed, on="other_entity_id")

    feature_rows = []
    for row in merged.itertuples():
        feats = build_pair_features(
            row.name_a, row.addr_a, row.country_a,
            row.name_b, row.addr_b, row.country_b,
        )
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    scores = model.predict_proba(feat_df[feature_columns])[:, 1]
    return pd.DataFrame({
        "source1_entity_id": merged["source1_entity_id"].values,
        "other_entity_id": merged["other_entity_id"].values,
        "score": scores,
    })


def assign_matches(scored_df: pd.DataFrame, threshold: float) -> dict:
    result = {}
    for s1_id, group in scored_df.groupby("source1_entity_id"):
        matched = group[group["score"] >= threshold]["other_entity_id"].tolist()
        result[s1_id] = matched
    return result


def run_inference(dataset_dir: str, model_path: str, output_dir: str, use_embeddings: bool = True) -> None:
    s1_df = read_source_tsv(os.path.join(dataset_dir, "test_source1.tsv"))
    s2_df = read_source_tsv(os.path.join(dataset_dir, "test_source2.tsv"))
    s3_df = read_source_tsv(os.path.join(dataset_dir, "test_source3.tsv"))
    others_df = pd.concat([s2_df, s3_df], ignore_index=True)

    embedder = default_embedder if use_embeddings else None
    candidates = generate_candidates(s1_df, s2_df, s3_df, embedder=embedder)

    candidate_rows = candidates_to_rows(candidates)
    for entity_id in s1_df["entity_id"]:
        candidate_rows.setdefault(entity_id, [])
    write_id_mapping_tsv(
        candidate_rows, os.path.join(output_dir, "candidate_pairs.tsv"),
        id_col="source1_entity_id", list_col="candidate_entity_ids",
    )

    artifact = load_model_artifact(model_path)
    scored_df = score_candidates(s1_df, others_df, candidates, artifact["model"], artifact["feature_columns"])
    matches = assign_matches(scored_df, artifact["threshold"])

    for entity_id in s1_df["entity_id"]:
        matches.setdefault(entity_id, [])
    write_id_mapping_tsv(
        matches, os.path.join(output_dir, "matching_results.tsv"),
        id_col="source1_entity_id", list_col="matched_entity_ids",
    )
