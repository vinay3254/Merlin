import random

import pandas as pd
from lightgbm import LGBMClassifier

from src.features import build_pair_features
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
