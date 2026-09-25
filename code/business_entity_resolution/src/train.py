import os
import random

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.features import featurize_pairs
from src.metrics import macro_f_beta
from src.utils_io import parse_id_list, read_ground_truth, read_source_tsv


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

    return featurize_pairs(pairs_df, s1_df, others_df, embed_lookup=embed_lookup)


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


EXCLUDED_FEATURE_COLUMNS = {"source1_entity_id", "other_entity_id", "label", "embedding_cosine"}
# embedding_cosine is excluded deliberately: embedding-based blocking is not
# used by default (deferred — see the spec's "Scale redesign addendum"), so
# this column would be a constant 0.0 and add nothing but noise.


def run_training(dataset_dir: str, model_path: str, use_embeddings: bool = False, val_frac: float = 0.2, seed: int = 42) -> dict:
    """
    Full training orchestration: load train data, generate candidates via
    blocking, build training pairs, featurize, fit the model, tune the
    decision threshold on a held-out validation split, save the artifact.
    Returns a summary dict: {"threshold": float, "n_train_pairs": int,
    "n_val_entities": int, "val_f_beta": float}.
    """
    # Imported lazily to avoid a circular import: src.infer imports from
    # src.train (load_model_artifact), so src.train cannot import src.infer
    # at module load time.
    from src.infer import generate_candidates, score_candidates

    s1_df = read_source_tsv(os.path.join(dataset_dir, "train_source1.tsv"))
    s2_df = read_source_tsv(os.path.join(dataset_dir, "train_source2.tsv"))
    s3_df = read_source_tsv(os.path.join(dataset_dir, "train_source3.tsv"))
    ground_truth_df = read_ground_truth(os.path.join(dataset_dir, "train_ground_truth.tsv"))
    others_df = pd.concat([s2_df, s3_df], ignore_index=True)

    train_ids, val_ids = split_train_validation(s1_df, val_frac=val_frac, seed=seed)
    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids)]
    val_s1 = s1_df[s1_df["entity_id"].isin(val_ids)]

    # use_embeddings is accepted for future use but not wired to anything
    # yet: embedding-based blocking is deliberately deferred (real-scale
    # memory issues, sentence-transformers not installed) per the spec's
    # "Scale redesign addendum". Passing use_embeddings=True is a no-op,
    # not an error.
    embedder = None
    # Block once against the full s1_df rather than separately against
    # train_s1 and val_s1: the expensive part of blocking (tokenizing and
    # indexing others_df) is identical either way, so blocking per-split
    # would rebuild that shared index twice for no benefit. Split the
    # resulting candidates dict by id membership instead.
    all_candidates = generate_candidates(s1_df, s2_df, s3_df, embedder=embedder, others_df=others_df)
    train_candidates = {s1_id: all_candidates[s1_id] for s1_id in train_ids}
    val_candidates = {s1_id: all_candidates[s1_id] for s1_id in val_ids}

    train_gt = ground_truth_df[ground_truth_df["source1_entity_id"].isin(train_ids)]
    val_gt = ground_truth_df[ground_truth_df["source1_entity_id"].isin(val_ids)]

    pairs_df = build_training_pairs(train_s1, others_df, train_gt, train_candidates)
    feature_matrix = build_feature_matrix(pairs_df, train_s1, others_df)
    feature_columns = [c for c in feature_matrix.columns if c not in EXCLUDED_FEATURE_COLUMNS]

    model = train_model(feature_matrix, feature_columns)

    val_scored = score_candidates(val_s1, others_df, val_candidates, model, feature_columns)
    threshold = tune_threshold(val_scored, val_gt)

    save_model_artifact(model, threshold, feature_columns, model_path)

    val_gt_map = {row["source1_entity_id"]: set(parse_id_list(row["matched_entity_ids"])) for _, row in val_gt.iterrows()}
    above = val_scored[val_scored["score"] >= threshold]
    predictions = above.groupby("source1_entity_id")["other_entity_id"].agg(set).to_dict()
    predictions = {s1_id: predictions.get(s1_id, set()) for s1_id in val_gt_map}
    val_f_beta = macro_f_beta(predictions, val_gt_map)

    return {
        "threshold": threshold,
        "n_train_pairs": len(feature_matrix),
        "n_val_entities": len(val_ids),
        "val_f_beta": val_f_beta,
    }
