# Handoff — 2026-09-26 session

This document exists so whoever picks this up next (including future-you)
doesn't have to rediscover everything the hard way. It's a narrative, not a
spec — see `CLAUDE.md` for the current-state summary and `docs/superpowers/`
for the design spec and implementation plan.

## TL;DR

- Went from "no trained model, 0.6302 baseline validation F0.5" to a
  validated **0.8341 validation F0.5** model (`model_v2.joblib`), with test-set
  inference already run and passing the official validator.
- Getting there took **4 full-scale training attempts**; the first 3 crashed
  or had to be killed due to out-of-memory / swap-thrashing on this
  machine's 15GB RAM. Root causes and fixes are below — read this before
  touching blocking/training code again, several of the "obvious"
  improvements tried here are actually landmines at this dataset's scale
  (2.2M–10.3M rows per file).
- `model.joblib` (0.6302, no embeddings) is left untouched as a fallback.
  `model_v2.joblib` (0.8341, with embeddings) is the current best.

## What was actually wrong, and what fixed it

### 1. Embedding blocking was never wired up, and wasn't memory-safe

`src/blocking.py`'s `embedding_knn_candidates` (semantic kNN blocking via
sentence-transformers + FAISS) existed but `src/train.py`'s `run_training`
had `use_embeddings` as a documented no-op — the embedder was never actually
passed through. Separately, `_search_partition` (the function that does the
actual FAISS search per country) called the embedder on the **entire**
partition's texts in one shot (`embedder(_combined_texts(other_df))`) and
built an `IndexIVFFlat` (stores full float32 vectors). At this dataset's
scale the largest single country partition is ~6.2M rows, so that's
~6.2M × 384 dims × 4 bytes ≈ 9.5GB just for the vectors, before the index
itself — a straightforward way to OOM a 15GB machine, and exactly the kind
of "never build a dense matrix across a full source" mistake `CLAUDE.md`
already warns about, just at one level of indirection further in (FAISS
still needs the raw vectors to build a flat/IVFFlat index).

**Fix:** `default_embedder` now encodes in batches (`batch_size=256`, ~8200
texts/sec on this machine's RTX 4050 laptop GPU). `_search_partition` /
`_build_other_index` now encode+add in bounded chunks (never materializing
more than one chunk's worth of vectors at a time) and switch to
`IndexIVFPQ` (compressed product-quantization codes, ~48 bytes/vector
instead of ~1536) once a partition exceeds `PQ_INDEX_THRESHOLD = 1_000_000`
rows. `run_training` now actually passes `default_embedder` through when
`use_embeddings=True`.

### 2. Raising blocking recall via doc-freq caps looked free, wasn't

First recall-improvement attempt: raise `MAX_TOKEN_DOC_FREQ` /
`MAX_ADDRESS_PREFIX_DOC_FREQ` from 500 to 5000 (10x), reasoning that top_k
truncation happens after the join so it "only" affects which tokens
qualify. This is true per-batch, but `S1_BATCH_SIZE=50_000` and doc-freq cap
together bound the worst-case per-batch merge size
(`batch_size × max_doc_freq`). Raising the cap 10x without shrinking the
batch size 10x let a single common token blow a batch's merge up toward
hundreds of millions of rows — measurable on a 3000-row sample (looked
fine, because the whole sample fit in one batch and never exercised
`S1_BATCH_SIZE` at all) but not on the full 2.2M-row dataset (batches
actually kick in, batch count is what matters).

**Also tried:** shrinking `S1_BATCH_SIZE` proportionally
(`25_000_000 // max_doc_freq`) to keep the product bounded. This fixes the
memory risk but creates a **second**, non-memory problem:
`_weighted_top_k_candidates` re-filters the *entire* s1 key table with
`.isin()` once per batch, so 10x more batches means ~10x more total
filter-scan work — potentially hours of extra runtime, independent of
whether it fits in RAM.

**Fix / final state:** reverted `MAX_TOKEN_DOC_FREQ` /
`MAX_ADDRESS_PREFIX_DOC_FREQ` back to 500 and `S1_BATCH_SIZE` back to
50_000 — the exact values already proven safe and fast at full 2.2M-entity
scale by the very first successful run. Recall is won via
`embedding_knn_candidates` instead (semantic matching finds pairs
token/address overlap structurally can't, regardless of top_k).

### 3. The candidate dictionary itself is the real memory ceiling

This is the one to actually remember. `run_training` calls
`generate_candidates` once for the **full** 2.2M-entity `s1_df`, producing
one dict `{s1_id: set_of_candidate_ids}` covering every entity, merged
across all 3 blocking strategies (`union_candidates`). That dict has to
live in RAM for a meaningful chunk of the run.

Measured directly (see `test_candidate_dict_mem.py`-style synthetic
benchmark — not committed, but trivial to reproduce): holding this dict for
all 2,206,822 entities costs **~12.76GB** at ~55 average unique candidates
per entity (roughly what `TOP_K_PER_STRATEGY=50` + `embedding top_k=20`
produces), and **~10.69GB** at ~40 average (what `TOP_K_PER_STRATEGY=25` +
`embedding top_k=8` produces). This is *before* the raw source dataframes,
FAISS index memory, or feature-building overhead. It's a pure function of
"2.2M entities × N candidate ID strings × ~70-90 bytes per string-in-a-set",
and no amount of clever batching inside the blocking functions changes it,
because the *return value itself* is one dict for every entity.

**What actually fixed the OOM (combination of things, all in
`src/train.py` / `src/blocking.py`):**

- Trimmed `TOP_K_PER_STRATEGY` to 25 and embedding `top_k` to 8 (down from
  an earlier, untested bump to 50 / 20) — directly shrinks the dict.
- `run_training` now explicitly `del`s `s2_df, s3_df` right after
  `pd.concat`-ing them into `others_df` — `pd.concat` copies data, so
  keeping the originals around duplicates ~7GB of business-name/address
  text for no reason. (`generate_candidates` ignores its `s2_df`/`s3_df`
  params whenever `others_df` is passed explicitly, which `run_training`
  always does — so this is a pure win, not a behavior change.)
- `run_training` now explicitly `del`s `all_candidates` right after
  splitting it into `train_candidates`/`val_candidates`, and `del`s
  `train_candidates` right after `build_training_pairs` consumes it (with
  a `gc.collect()` after) — freeing the ~80% training-entity share of the
  dict before the memory-heavy feature-build step, since only
  `val_candidates` is needed after that point.

None of this is a strict data-structure fix (e.g. interning entity ID
strings, or representing candidates as integer arrays instead of Python
sets of strings) — that would shrink the ceiling further but wasn't
necessary once these were combined with the trimmed top_k values. If future
recall-tuning re-hits this wall, that's the next lever to pull, not another
top_k bump.

### 4. `featurize_pairs` and `build_training_pairs` built a Python dict per row

Both functions used to do the equivalent of
`rows.append({"col1": v1, "col2": v2, ...})` in a loop, once per pair. At
this dataset's scale (27–33 million pairs), a dict costs several hundred
bytes of pure Python object overhead per row — multiple GB of overhead that
has nothing to do with the actual feature values, and (in
`featurize_pairs`'s case) accumulated across *all* chunks
(`chunk_frames.append(feat_df)`) until one final `pd.concat` at the end.

Measured before/after on a synthetic 3-million-pair benchmark: **before**,
this pattern was directly implicated in a run that reached ~13GB RSS with
GPU idle and swap climbing toward 9GB+ (had to be killed). **After**
rewriting both functions to build flat per-column lists instead (append
scalars to `feature_columns[key]` instead of building a dict per row, same
idea in `build_training_pairs` with `s1_ids`/`other_ids`/`labels` lists):
3 million pairs processed in 75.8s at a peak RSS of **1.77GB** — roughly
39,600 pairs/sec, comfortably fitting a 27–33M-pair full run in a few GB
and ~15-20 minutes.

### 5. Added a candidates cache to make further iteration cheap

Not a bug fix, but a deliberate infra addition once `model_v2.joblib`
landed: `run_training` now takes `candidates_cache_path` (CLI:
`--candidates-cache <path>`). On a cache miss it runs blocking as normal
and then `joblib.dump`s the train/val candidate dicts to that path before
consuming them; on a cache hit it loads them instead and skips
`generate_candidates` entirely. Also added `--max-negatives-per-positive`
(previously hardcoded at 10) so negative-sampling can be swept without
code changes.

**This has not been run yet** — added but untested beyond a quick
`pytest` pass (79 tests still green) and was mid-way through a small-scale
(`--max-train-entities 2000`) smoke test when the session paused. Before
trusting it on a real run: confirm the cache round-trip actually produces
the same `val_f_beta` as a no-cache run on the same small sample, and check
`joblib.dump`'s output file size for the full 2.2M-entity candidate dict
(expect several GB — `df -h` showed 26GB free on this machine at the time
of writing, should be enough headroom).

Intent: pay the ~2 hour blocking cost once with `--use-embeddings
--candidates-cache candidates_v1.pkl`, then run many `~20-30` min
classifier/feature/negative-sampling experiments reusing that same cache
path, rather than re-paying the blocking cost per experiment.

## What a full run actually looks like now

A full `train_cli.py --use-embeddings` run on `student_resource/dataset/train`
(2,206,822 S1 entities, 10,320,220 combined S2+S3 rows) takes **~2 hours**,
almost entirely GPU-encoding time: the embedding step encodes the full
"other" corpus once per country partition (~6.2M for US, ~4.1M for India),
plus the full S1 side, at ~8200 texts/sec batched on an RTX 4050 laptop
GPU (6GB VRAM). Everything after blocking (candidate-pair building,
feature-building, LightGBM fit, threshold tuning) now takes on the order of
15-25 minutes total, not hours — that part was never the bottleneck once
the dict-per-row issue was fixed.

Peak RSS during a real run sits around 10-13GB out of 15GB total, with
swap usage in the low single-digit GB — tight, but survives. **Watch `free
-h` and the process's GPU utilization (`nvidia-smi`) while a full run is
going**, not just whether it's still alive: a GPU utilization drop to 0%
combined with climbing RSS/swap and non-trivial CPU usage is the signature
of the feature-building or candidate-dict-construction phases, not a hang —
but a *rapid* swap climb (tens of MB/s sustained, visible via `vmstat 1`)
combined with available RAM near zero is the signature of real thrashing
and is worth intervening on. The user's explicit guidance this session: the
machine has 31GB of swap and it's there to be used — don't kill a run just
because RAM is "tight", only if it's genuinely heading toward exhausting
both RAM and swap.

## Results on record

| Model | Blocking | Val F0.5 | Threshold | Training pairs | Val entities |
|---|---|---|---|---|---|
| `model.joblib` | token+address only | 0.6302 | 0.75 | 27,264,472 | 441,364 |
| `model_v2.joblib` | token+address+embeddings | **0.8341** | 0.70 | 33,212,774 | 441,364 |

Both are full-scale runs on the real training set (not samples). A small
diagnostic (3000-entity sample) run partway through this session, at
slightly different top_k settings, predicted ~0.813 for the embeddings
config — the real 0.8341 full-scale result is consistent with that.

Blocking recall is still not 100%: the same small-sample diagnostic found
~79.5% of true positive pairs actually reachable by the union of all 3
blocking strategies (oracle ceiling ~0.907 macro F0.5 if the classifier
were perfect). That ~20% recall gap is the most direct lever left to
improve beyond 0.8341, ahead of classifier/feature tuning.

Test-set inference has been run with `model_v2.joblib`
(`--use-embeddings`, matching how it was trained) and written to
`output/matching_results.tsv` + `output/candidate_pairs.tsv`. Both passed
`validate_submission.py` (`PASS — no blocking issues found. Safe to
submit.`). **Leaderboard submission status is not confirmed as of writing
this document** — check before assuming it happened.

Reference leaderboard context (from the Unstop hackathon page, screenshot
shared mid-session): 1st place was at 0.984098, 2nd at 0.98377, 3rd at
0.983149. 0.8341 is a large improvement over the 0.6302 starting point but
is not yet competitive with the top of that leaderboard.

## If you're picking this up next

1. Read the comments in `src/blocking.py` (top of file, and around
   `TOP_K_PER_STRATEGY` / `embedding_knn_candidates`) and `src/train.py`
   (around `run_training`'s `del`/`gc.collect()` calls) before changing any
   blocking constants. They explain *why* those specific values, not just
   what they are.
2. If you want to push recall further, the candidate-dict ceiling
   (section 3 above) is the thing to budget for. Either accept the current
   top_k values, or invest in a real fix (batch the whole pipeline by
   outer s1-entity chunks so the full-scale candidate dict never exists at
   once, or represent candidate IDs as integers via a global id→int mapping
   instead of Python strings in sets) before raising top_k again.
3. `student_resource/Documentation_template.md` still needs to be filled in
   before a top-100 finish would require submitting methodology/source.
   Nothing has been written there yet.
4. Confirm whether `output/matching_results.tsv` /
   `output/candidate_pairs.tsv` were actually uploaded to the leaderboard.
5. Before relying on `--candidates-cache` (section 5 above) for real
   experiments, first verify it on a small `--max-train-entities` sample:
   run once to populate the cache, run again against the same cache path
   with a different `--max-negatives-per-positive`, and confirm both runs
   produce sane (not necessarily identical, since negative sampling differs)
   `val_f_beta` values and that the second run actually skips blocking
   (should be seconds, not minutes, for a 2000-entity sample). This was
   *not* verified before the session paused.
