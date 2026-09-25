import os
import random

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.features import build_pair_features
from src.metrics import macro_f_beta
from src.utils_io import parse_id_list


def build_training_pairs(
    s1_df, others_df, ground_truth_df, blocking_candidates,
    max_negatives_per_positive: int = 10, seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    gt_map = {
        s1_id: set(parse_id_list(matched))
        for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"])
    }

    rows = []
    for s1_id, positives in gt_map.items():
        negatives_pool = list(set(blocking_candidates.get(s1_id, set())) - positives)
        cap = max_negatives_per_positive * max(1, len(positives))
        if len(negatives_pool) > cap:
            negatives_pool = rng.sample(negatives_pool, cap)

        for other_id in positives:
            rows.append({"source1_entity_id": s1_id, "other_entity_id": other_id, "label": 1})
        for other_id in negatives_pool:
            rows.append({"source1_entity_id": s1_id, "other_entity_id": other_id, "label": 0})
    return pd.DataFrame(rows, columns=["source1_entity_id", "other_entity_id", "label"])


def build_feature_matrix(pairs_df, s1_df, others_df, embed_lookup=None) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "label"])

    s1_renamed = s1_df.rename(columns={
        "entity_id": "source1_entity_id", "business_name": "name_a",
        "business_address": "addr_a", "country": "country_a",
    })[["source1_entity_id", "name_a", "addr_a", "country_a"]]
    others_renamed = others_df.rename(columns={
        "entity_id": "other_entity_id", "business_name": "name_b",
        "business_address": "addr_b", "country": "country_b",
    })[["other_entity_id", "name_b", "addr_b", "country_b"]]

    merged = pairs_df.merge(s1_renamed, on="source1_entity_id").merge(others_renamed, on="other_entity_id")

    feature_rows = []
    for row in merged.itertuples():
        embed_a = embed_lookup.get(row.source1_entity_id) if embed_lookup else None
        embed_b = embed_lookup.get(row.other_entity_id) if embed_lookup else None
        feats = build_pair_features(
            row.name_a, row.addr_a, row.country_a,
            row.name_b, row.addr_b, row.country_b,
            embed_a=embed_a, embed_b=embed_b,
        )
        feats["source1_entity_id"] = row.source1_entity_id
        feats["other_entity_id"] = row.other_entity_id
        feats["label"] = row.label
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)


def train_model(feature_matrix_df, feature_columns) -> LGBMClassifier:
    model = LGBMClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(feature_matrix_df[feature_columns], feature_matrix_df["label"])
    return model


def split_train_validation(s1_df, val_frac: float = 0.2, seed: int = 42):
    ids = list(s1_df["entity_id"])
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    val_size = max(1, round(len(ids) * val_frac)) if len(ids) >= 2 else 0
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return train_ids, val_ids


def tune_threshold(scored_pairs_df, ground_truth_df, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    gt_map = {
        s1_id: set(parse_id_list(matched))
        for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"])
    }

    best_threshold = thresholds[0]
    best_score = -1.0
    for threshold in thresholds:
        above = scored_pairs_df[scored_pairs_df["score"] >= threshold]
        grouped = above.groupby("source1_entity_id")["other_entity_id"].agg(set).to_dict()
        predictions = {s1_id: grouped.get(s1_id, set()) for s1_id in gt_map}
        score = macro_f_beta(predictions, gt_map)
        if score >= best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def save_model_artifact(model, threshold: float, feature_columns: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": feature_columns}, path)


def load_model_artifact(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model artifact found at {path}. Run train.py first.")
    return joblib.load(path)
