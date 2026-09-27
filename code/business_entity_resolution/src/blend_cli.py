"""
Blend LightGBM scores with cross-encoder scores on uncertain pairs.

  fit:   on the LightGBM model's saved validation pairs (<model>.val.npz),
         cross-encoder-score the pairs with lo < score < hi, fit a logistic
         blend of logit(lgb) and logit(ce), and pick the F0.5-best threshold
         with the one-S1-per-record assignment. Reports baseline vs blend.
  apply: same blend on the test scores.tsv.gz, writes matching_results.tsv.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from src.cross_encoder import log, read_rows, score_texts
from src.infer import assign_matches
from src.metrics import macro_f_beta
from src.utils_io import write_id_mapping_tsv


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def blend(lgb, ce, params):
    """Logistic blend of logit(lgb) and the logits of one or more cross-encoders (ce: (n,) or (n, k))."""
    ce = np.asarray(ce)
    if ce.ndim == 1:
        ce = ce[:, None]
    z = params["coef"][0] * _logit(lgb) + params["intercept"]
    for i in range(ce.shape[1]):
        z = z + params["coef"][i + 1] * _logit(ce[:, i])
    return 1 / (1 + np.exp(-z))


def score_models(model_dirs: str, text_a, text_b, caches: str = ""):
    """
    (n, k) scores from k comma-separated cross-encoder dirs. `caches` is an
    optional comma-separated list of .npy files aligned with model_dirs
    (empty entry = score with the model); a cached file must hold scores for
    exactly these pairs in this order.
    """
    dirs = [d for d in model_dirs.split(",") if d]
    cache_list = (caches.split(",") if caches else []) + [""] * len(dirs)
    cols = []
    for d, c in zip(dirs, cache_list):
        if c:
            arr = np.load(c)
            arr = arr[:, 0] if arr.ndim == 2 else arr
            assert len(arr) == len(text_a), f"cache {c} has {len(arr)} scores for {len(text_a)} pairs"
            cols.append(arr)
            log(f"loaded cached scores {c}")
        else:
            cols.append(score_texts(d, text_a, text_b))
            log(f"scored with {d}")
    return np.column_stack(cols)


def _ids(path):
    return pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"].to_numpy()


def _sweep(frame, truth, thresholds):
    results = []
    for t in thresholds:
        m = assign_matches(frame, t)
        results.append((macro_f_beta({k: set(v) for k, v in m.items()}, truth), float(t)))
    return results


def cmd_fit(args):
    from sklearn.linear_model import LogisticRegression

    d = args.dataset_dir
    z = np.load(args.val_scores)
    s1_ids = _ids(os.path.join(d, "train_source1.tsv"))
    other_paths = [os.path.join(d, "train_source2.tsv"), os.path.join(d, "train_source3.tsv")]
    other_ids = np.concatenate([_ids(p) for p in other_paths])
    gt = pd.read_csv(os.path.join(d, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    val_ent_ids = set(s1_ids[z["val_ents"]])
    truth = {s: set(m.split(",")) - {""} for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if s in val_ent_ids}
    for e in val_ent_ids:
        truth.setdefault(e, set())

    frame = pd.DataFrame({
        "source1_entity_id": s1_ids[z["s1_pos"]], "other_entity_id": other_ids[z["other_pos"]],
        "score": z["score"], "label": z["label"],
    })
    unsure = np.flatnonzero((frame["score"] > args.lo) & (frame["score"] < args.hi))
    log(f"val pairs {len(frame)}, uncertain {len(unsure)}")
    s1_text = read_rows([os.path.join(d, "train_source1.tsv")], z["s1_pos"][unsure])
    o_text = read_rows(other_paths, z["other_pos"][unsure])
    ce = score_models(args.model_dir, [s1_text[int(p)] for p in z["s1_pos"][unsure]],
                      [o_text[int(p)] for p in z["other_pos"][unsure]])
    lgb = frame["score"].to_numpy()[unsure]
    y = frame["label"].to_numpy()[unsure]
    reg = LogisticRegression(C=10.0).fit(np.column_stack([_logit(lgb)] + [_logit(ce[:, i]) for i in range(ce.shape[1])]), y)
    params = {"coef": reg.coef_[0].tolist(), "intercept": float(reg.intercept_[0]), "lo": args.lo, "hi": args.hi}
    log(f"blend params {params}")
    log(f"uncertain-pair accuracy: lgb {((lgb >= 0.5) == y).mean():.4f} "
        f"ce {[round(float(((ce[:, i] >= 0.5) == y).mean()), 4) for i in range(ce.shape[1])]} "
        f"blend {((blend(lgb, ce, params) >= 0.5) == y).mean():.4f}")

    thresholds = np.arange(0.3, 0.91, 0.05)
    base = _sweep(frame, truth, thresholds)
    blended = frame.copy()
    blended.loc[blended.index[unsure], "score"] = blend(lgb, ce, params)
    new = _sweep(blended, truth, thresholds)
    for (fb, t), (fn, _) in zip(base, new):
        log(f"threshold {t:.2f}: base F0.5 {fb:.4f}  blend F0.5 {fn:.4f}")
    best_base, best = max(base), max(new)
    params["threshold"] = best[1]
    params["val_f_base"], params["val_f_blend"] = best_base[0], best[0]
    log(f"best base {best_base}  best blend {best}")
    with open(args.params_out, "w") as f:
        json.dump(params, f, indent=2)


def cmd_apply(args):
    d = args.dataset_dir
    with open(args.params) as f:
        params = json.load(f)
    # Categorical ids: ~70M rows of Python id strings would take ~10GB.
    scored = pd.read_csv(os.path.join(args.scores_dir, "scores.tsv.gz"), sep="\t", keep_default_na=False,
                         dtype={"source1_entity_id": "category", "other_entity_id": "category", "score": np.float32})
    unsure = np.flatnonzero((scored["score"] > params["lo"]) & (scored["score"] < params["hi"]))
    log(f"test pairs {len(scored)}, uncertain {len(unsure)}")
    s1_path = os.path.join(d, "test_source1.tsv")
    other_paths = [os.path.join(d, "test_source2.tsv"), os.path.join(d, "test_source3.tsv")]
    s1_ids = _ids(s1_path)
    other_ids = np.concatenate([_ids(p) for p in other_paths])
    s1_index = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    o_index = pd.Series(np.arange(len(other_ids)), index=other_ids)
    sp = s1_index[scored["source1_entity_id"].to_numpy()[unsure].astype(str)].to_numpy()
    op = o_index[scored["other_entity_id"].to_numpy()[unsure].astype(str)].to_numpy()
    s1_text = read_rows([s1_path], sp)
    o_text = read_rows(other_paths, op)
    log("texts loaded")
    ce = score_models(args.model_dir, [s1_text[int(p)] for p in sp], [o_text[int(p)] for p in op], args.ce_cache)
    del s1_text, o_text
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "ce_scores_uncertain.npy"), ce)
    scores = scored["score"].to_numpy().copy()
    scores[unsure] = blend(scores[unsure], ce, params)
    scored["score"] = scores
    threshold = args.threshold if args.threshold is not None else params["threshold"]
    matches = assign_matches(scored, threshold)
    matches = {str(k): [str(v) for v in vs] for k, vs in matches.items()}
    for eid in s1_ids:
        matches.setdefault(eid, [])
    os.makedirs(args.output_dir, exist_ok=True)
    write_id_mapping_tsv(matches, os.path.join(args.output_dir, "matching_results.tsv"),
                         id_col="source1_entity_id", list_col="matched_entity_ids")
    shutil.copy(os.path.join(args.scores_dir, "candidate_pairs.tsv"), os.path.join(args.output_dir, "candidate_pairs.tsv"))
    log(f"wrote {args.output_dir} at threshold {threshold}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dataset-dir", required=True)
    f.add_argument("--val-scores", required=True)
    f.add_argument("--model-dir", required=True)
    f.add_argument("--params-out", required=True)
    f.add_argument("--lo", type=float, default=0.02)
    f.add_argument("--hi", type=float, default=0.98)
    a = sub.add_parser("apply")
    a.add_argument("--dataset-dir", required=True)
    a.add_argument("--scores-dir", required=True)
    a.add_argument("--model-dir", required=True)
    a.add_argument("--params", required=True)
    a.add_argument("--output-dir", required=True)
    a.add_argument("--threshold", type=float, default=None)
    a.add_argument("--ce-cache", default="", help="comma list of .npy score caches aligned with --model-dir")
    args = parser.parse_args(argv)
    {"fit": cmd_fit, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
