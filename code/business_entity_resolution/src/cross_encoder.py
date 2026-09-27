"""
Cross-encoder re-scoring for uncertain pairs.

A multilingual MiniLM (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
Apache-2.0, ~118M params) is fine-tuned as a binary pair classifier on
"name | address" text of both records. It reads both records jointly, so it
can learn generator patterns string features miss (transliterated Indic
names, junk brand names over a copied address, which number edits mean a
different business). Too slow to run on every candidate, so it only
re-scores pairs the LightGBM model is unsure about.

  python -m src.cross_encoder prepare --dataset-dir DIR --candidates cands_train.npz --out-dir ce/
  python -m src.cross_encoder train --data-dir ce/ --model-dir ce/model
  python -m src.cross_encoder score --model-dir ce/model --pairs pairs.tsv --out scores.npy

Memory: source files are streamed in chunks and only the needed rows kept,
so preparation runs alongside other jobs on a 15GB machine.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 128
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def record_text(name, address) -> str:
    return f"{name} | {address}".strip()


def read_rows(paths, wanted_positions):
    """
    Stream the concatenation of `paths` (as read_source_tsv would order it)
    and return {position: text} for the wanted row positions only.
    """
    wanted = np.unique(np.asarray(wanted_positions, dtype=np.int64))
    out = {}
    offset = 0
    for path in paths:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
            chunk = chunk.fillna("")
            lo, hi = offset, offset + len(chunk)
            sel = wanted[(wanted >= lo) & (wanted < hi)]
            for pos in sel:
                row = chunk.iloc[pos - lo]
                out[int(pos)] = record_text(row["business_name"], row["business_address"])
            offset = hi
    return out


def cmd_prepare(args):
    from src.train import split_train_validation

    d = args.dataset_dir
    s1_ids = pd.read_csv(os.path.join(d, "train_source1.tsv"), sep="\t", dtype=str, usecols=["entity_id"])["entity_id"]
    s1_df = pd.DataFrame({"entity_id": s1_ids})
    train_ids, val_ids = split_train_validation(s1_df)
    pos_of = pd.Series(np.arange(len(s1_ids)), index=s1_ids.to_numpy())
    rng = random.Random(args.seed)
    splits = {
        "train": pos_of[rng.sample(train_ids, args.train_entities)].to_numpy(),
        "val": pos_of[rng.sample(val_ids, args.val_entities)].to_numpy(),
    }

    z = np.load(args.candidates)
    s1_pos, other_pos, blk = z["s1_pos"].astype(np.int64), z["other_pos"].astype(np.int64), z["score"]
    other_paths = [os.path.join(d, "train_source2.tsv"), os.path.join(d, "train_source3.tsv")]
    other_ids = pd.concat(
        [pd.read_csv(p, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"] for p in other_paths],
        ignore_index=True,
    ).to_numpy()
    gt = pd.read_csv(os.path.join(d, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    pairs = gt.assign(o=gt["matched_entity_ids"].str.split(",")).explode("o")
    pairs = pairs[pairs["o"] != ""]
    other_index = pd.Series(np.arange(len(other_ids)), index=other_ids)
    true_keys = pos_of[pairs["source1_entity_id"].to_numpy()].to_numpy().astype(np.int64) * len(other_ids) + \
        other_index[pairs["o"].to_numpy()].to_numpy()
    del gt, pairs
    log("loaded candidates and ground truth")

    selected = {}
    for name, ents in splits.items():
        rows = np.flatnonzero(np.isin(s1_pos, ents))
        frame = pd.DataFrame({"s": s1_pos[rows], "o": other_pos[rows], "blk": blk[rows]})
        frame["label"] = np.isin(frame["s"].to_numpy() * len(other_ids) + frame["o"].to_numpy(), true_keys)
        frame["rank"] = frame.groupby("s")["blk"].rank(ascending=False, method="first")
        # All positives + the hardest negatives (highest blocking similarity).
        keep = frame["label"] | (frame["rank"] <= args.negatives_per_entity + 4)
        frame = frame[keep]
        neg = frame[~frame["label"]].groupby("s").head(args.negatives_per_entity)
        selected[name] = pd.concat([frame[frame["label"]], neg]).sample(frac=1.0, random_state=args.seed)
        log(f"{name}: {len(selected[name])} pairs ({int(selected[name]['label'].sum())} positive)")
    del s1_pos, other_pos, blk

    s1_text = read_rows([os.path.join(d, "train_source1.tsv")], np.concatenate([f["s"] for f in selected.values()]))
    o_text = read_rows(other_paths, np.concatenate([f["o"] for f in selected.values()]))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, frame in selected.items():
        out = pd.DataFrame({
            "text_a": [s1_text[s] for s in frame["s"]],
            "text_b": [o_text[o] for o in frame["o"]],
            "label": frame["label"].astype(int).to_numpy(),
        })
        out.to_csv(os.path.join(args.out_dir, f"{name}.tsv"), sep="\t", index=False)
    log("prepared")


class PairDataset:
    def __init__(self, text_a, text_b, labels=None):
        self.a, self.b, self.y = list(text_a), list(text_b), labels

    def __len__(self):
        return len(self.a)


def _batches(tokenizer, a, b, y, batch_size, shuffle, seed=0):
    import torch

    idx = np.arange(len(a))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        enc = tokenizer([a[i] for i in sel], [b[i] for i in sel], truncation=True,
                        max_length=MAX_LEN, padding=True, return_tensors="pt")
        labels = None if y is None else torch.tensor(y[sel], dtype=torch.float32)
        yield enc, labels


def cmd_train(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    tr = pd.read_csv(os.path.join(args.data_dir, "train.tsv"), sep="\t", keep_default_na=False)
    va = pd.read_csv(os.path.join(args.data_dir, "val.tsv"), sep="\t", keep_default_na=False)
    if args.max_train:
        tr = tr.head(args.max_train)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = (len(tr) + args.batch_size - 1) // args.batch_size
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    a, b, y = tr["text_a"].tolist(), tr["text_b"].tolist(), tr["label"].to_numpy()
    model.train()
    for step, (enc, labels) in enumerate(_batches(tok, a, b, y, args.batch_size, shuffle=True)):
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
        loss = loss_fn(logits.float(), labels.cuda())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            log(f"step {step}/{steps} loss {loss.item():.4f}")
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(args.model_dir)
    tok.save_pretrained(args.model_dir)
    probs = score_texts(args.model_dir, va["text_a"].tolist(), va["text_b"].tolist(), model=model, tokenizer=tok)
    yv = va["label"].to_numpy()
    acc = ((probs >= 0.5) == yv).mean()
    ll = -np.mean(yv * np.log(np.clip(probs, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - probs, 1e-6, 1)))
    log(f"val accuracy {acc:.4f} logloss {ll:.4f} on {len(va)} pairs")


def score_texts(model_dir, text_a, text_b, model=None, tokenizer=None, batch_size=512):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()
    model.eval()
    out = []
    with torch.no_grad():
        for enc, _ in _batches(tokenizer, text_a, text_b, None, batch_size, shuffle=False):
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.squeeze(-1)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def cmd_score(args):
    pairs = pd.read_csv(args.pairs, sep="\t", keep_default_na=False)
    log(f"scoring {len(pairs)} pairs")
    probs = score_texts(args.model_dir, pairs["text_a"].tolist(), pairs["text_b"].tolist())
    np.save(args.out, probs.astype(np.float32))
    log(f"saved {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-entities", type=int, default=100_000)
    p.add_argument("--val-entities", type=int, default=5_000)
    p.add_argument("--negatives-per-entity", type=int, default=6)
    p.add_argument("--seed", type=int, default=11)
    t = sub.add_parser("train")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--model-dir", required=True)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--max-train", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--pairs", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"prepare": cmd_prepare, "train": cmd_train, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
