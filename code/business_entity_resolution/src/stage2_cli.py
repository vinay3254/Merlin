"""
Stage 2: sibling-consistency re-scoring.

A true match usually resembles the S1 entity's OTHER confident matches (the
S2/S3 records of one business are variants of each other), while a hard
negative (same street, different business; same name, different branch)
often does not. For each candidate pair this computes the best name /
address similarity to the entity's other confident candidates (blended
score >= CONF), and a small LightGBM re-scores [logit(score), sibling
features]. Measured on v5 validation (2-fold by entity): F0.5 0.9627 ->
0.9649 on LightGBM scores alone.

  fit:   blended validation scores (LightGBM + cross-encoder on uncertain
         pairs) -> sibling features -> stage-2 model (+ 2-fold F report).
  apply: blended test scores from blend_cli apply -> stage-2 -> outputs.
"""
import argparse
import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.blend_cli import _ids, _logit, blend
from src.cross_encoder import log, read_rows, score_texts
from src.infer import assign_matches
from src.metrics import macro_f_beta
from src.normalize_v3 import address_tokens, name_parts
from src.utils_io import write_id_mapping_tsv

CONF = 0.9
FLOOR = 0.01  # pairs below this keep their score; they are never assigned anyway
COLS = ["logit", "sib_name", "sib_addr", "sib_max", "n_conf"]


def _norm(text):
    name, _, addr = text.partition(" | ")
    return " ".join(name_parts(name)[0]), " ".join(address_tokens(addr))


def sibling_features(s_key, o_key, score, other_text):
    """
    s_key / o_key: per-pair integer keys of the S1 entity and other record;
    score: per-pair probability; other_text: {o_key: "name | address"}.
    Returns a DataFrame of COLS aligned with the inputs.
    """
    n = len(s_key)
    feats = {c: np.zeros(n, np.float32) for c in ("sib_name", "sib_addr", "sib_max", "n_conf")}
    norm = {}
    active = np.flatnonzero(score >= FLOOR)
    groups = pd.Series(active).groupby(s_key[active]).indices
    for idx in groups.values():
        grp = active[idx]
        conf = grp[score[grp] >= CONF]
        feats["n_conf"][grp] = len(conf)
        if len(conf) == 0:
            continue
        for i in grp:
            if o_key[i] not in norm:
                norm[o_key[i]] = _norm(other_text[o_key[i]])
        cand = [norm[o_key[i]] for i in grp]
        confn = [norm[o_key[i]] for i in conf]
        mn = process.cdist([c[0] for c in cand], [c[0] for c in confn], scorer=fuzz.token_set_ratio)
        ma = process.cdist([c[1] for c in cand], [c[1] for c in confn], scorer=fuzz.token_set_ratio)
        self_col = {c: j for j, c in enumerate(conf)}
        for r, i in enumerate(grp):
            j = self_col.get(i)
            if j is not None:
                mn[r, j] = -1
                ma[r, j] = -1
        feats["sib_name"][grp] = mn.max(axis=1)
        feats["sib_addr"][grp] = ma.max(axis=1)
        feats["sib_max"][grp] = ((mn + ma) / 2).max(axis=1)
    out = pd.DataFrame(feats)
    out["logit"] = _logit(score)
    return out[COLS]


def cmd_fit(args):
    import lightgbm as lgb

    d = args.dataset_dir
    with open(args.blend_params) as f:
        params = json.load(f)
    z = np.load(args.val_scores)
    s1_ids = _ids(os.path.join(d, "train_source1.tsv"))
    other_paths = [os.path.join(d, "train_source2.tsv"), os.path.join(d, "train_source3.tsv")]
    other_ids = np.concatenate([_ids(p) for p in other_paths])
    gt = pd.read_csv(os.path.join(d, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    val_ent_ids = set(s1_ids[z["val_ents"]])
    truth = {s: set(m.split(",")) - {""} for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if s in val_ent_ids}
    for e in val_ent_ids:
        truth.setdefault(e, set())
    g = gt.assign(x=gt["matched_entity_ids"].str.split(",")).explode("x")
    owner = dict(zip(g["x"], g["source1_entity_id"]))
    del gt, g

    f = pd.DataFrame({"s": z["s1_pos"], "o": z["other_pos"], "score": z["score"].astype(np.float64), "label": z["label"]})
    f["source1_entity_id"] = s1_ids[f["s"]]
    f["other_entity_id"] = other_ids[f["o"]]
    # Drop false positives whose true owner is outside the validation sample:
    # on test every S1 entity competes, so those records would go to their owner.
    f = f[~(f["other_entity_id"].map(owner).notna() & ~f["label"])].reset_index(drop=True)
    unsure = np.flatnonzero((f["score"] > params["lo"]) & (f["score"] < params["hi"]))
    s1_text = read_rows([os.path.join(d, "train_source1.tsv")], f["s"].to_numpy()[unsure])
    o_text = read_rows(other_paths, f["o"].unique())
    ce = score_texts(args.model_dir, [s1_text[int(p)] for p in f["s"].to_numpy()[unsure]],
                     [o_text[int(p)] for p in f["o"].to_numpy()[unsure]])
    blended = f["score"].to_numpy().copy()
    blended[unsure] = blend(blended[unsure], ce, params)
    f["score"] = blended
    log(f"val pairs {len(f)}, uncertain re-scored {len(unsure)}")

    x = sibling_features(f["s"].to_numpy(), f["o"].to_numpy(), f["score"].to_numpy(), o_text)
    log("sibling features built")
    y = f["label"].to_numpy()
    ents = f["s"].unique()
    fold = pd.Series(np.random.default_rng(0).integers(0, 2, len(ents)), index=ents)[f["s"]].to_numpy()
    s2 = np.zeros(len(f))
    make = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1)
    for k in (0, 1):
        m = make().fit(x[fold != k], y[fold != k])
        s2[fold == k] = m.predict_proba(x[fold == k])[:, 1]
    best = (0.0, 0.5)
    for t in np.arange(0.4, 0.91, 0.05):
        fb = macro_f_beta({k: set(v) for k, v in assign_matches(f, t).items()}, truth)
        fs = macro_f_beta({k: set(v) for k, v in assign_matches(f.assign(score=s2), t).items()}, truth)
        log(f"threshold {t:.2f}: blend F0.5 {fb:.4f}  stage2 F0.5 {fs:.4f}")
        best = max(best, (fs, float(t)))
    log(f"best stage2 {best}")
    model = make().fit(x, y)
    joblib.dump({"model": model, "threshold": best[1], "val_f": best[0], "blend_params": params}, args.model_out)
    log(f"saved {args.model_out}")


def cmd_apply(args):
    d = args.dataset_dir
    art = joblib.load(args.stage2_model)
    params = art["blend_params"]
    scored = pd.read_csv(os.path.join(args.scores_dir, "scores.tsv.gz"), sep="\t", keep_default_na=False,
                         dtype={"source1_entity_id": "category", "other_entity_id": "category", "score": np.float32})
    # Same float32 comparison as blend_cli apply, so the mask matches exactly.
    unsure = np.flatnonzero((scored["score"] > params["lo"]) & (scored["score"] < params["hi"]))
    scores = scored["score"].to_numpy().astype(np.float64)
    ce = np.load(os.path.join(args.ce_dir, "ce_scores_uncertain.npy"))
    assert len(ce) == len(unsure), "cross-encoder scores do not match the uncertain-pair mask"
    scores[unsure] = blend(scores[unsure], ce, params)
    log(f"test pairs {len(scores)}, blended {len(unsure)}")

    other_paths = [os.path.join(d, "test_source2.tsv"), os.path.join(d, "test_source3.tsv")]
    other_ids = np.concatenate([_ids(p) for p in other_paths])
    o_index = pd.Series(np.arange(len(other_ids)), index=other_ids)
    s_key = scored["source1_entity_id"].cat.codes.to_numpy()
    active = scores >= FLOOR
    o_key = np.full(len(scores), -1, dtype=np.int64)
    o_key[active] = o_index[scored["other_entity_id"].to_numpy()[active].astype(str)].to_numpy()
    o_text = read_rows(other_paths, np.unique(o_key[active]))
    log("texts loaded")
    x = sibling_features(s_key, o_key, scores, o_text)
    del o_text
    log("sibling features built")
    s2 = scores.copy()
    s2[active] = art["model"].predict_proba(x[active])[:, 1]
    scored["score"] = s2
    threshold = args.threshold if args.threshold is not None else art["threshold"]
    matches = assign_matches(scored, threshold)
    matches = {str(k): [str(v) for v in vs] for k, vs in matches.items()}
    for eid in _ids(os.path.join(d, "test_source1.tsv")):
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
    f.add_argument("--blend-params", required=True)
    f.add_argument("--model-out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--dataset-dir", required=True)
    a.add_argument("--scores-dir", required=True, help="dir with scores.tsv.gz and candidate_pairs.tsv")
    a.add_argument("--ce-dir", required=True, help="dir with ce_scores_uncertain.npy from blend_cli apply")
    a.add_argument("--stage2-model", required=True)
    a.add_argument("--output-dir", required=True)
    a.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    {"fit": cmd_fit, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
