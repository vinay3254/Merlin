# Handoff — 2026-09-26 session

This document exists so whoever picks this up next (including future-you)
doesn't have to rediscover everything the hard way. It's a narrative, not a
spec — see `CLAUDE.md` for the current-state summary and `docs/superpowers/`
for the design spec and implementation plan.

## Session 2026-09-26 evening → 2026-09-27 (v3 → v7): read this first

Everything below this section describes the older v2 pipeline and is kept
as history. This section is the current story.

### Where things stand

| version | what changed | realistic val F0.5* | public LB |
|---|---|---|---|
| v2 | pandas blocking + embeddings, LightGBM | 0.834 | 0.789 |
| v2 + rule | each S2/S3 record kept only for its best S1 entity (t=0.70) | – | 0.802 |
| v3 | sparse TF-IDF blocking, rapidfuzz features | ~0.952 | 0.930 |
| v3b | + global name/address context features | – | 0.934 |
| v3b + CE | + cross-encoder v1 on uncertain pairs | 0.969 | 0.943 |
| v5 + CE + stage 2 | multi-list blocking, number/skeleton features, region canonicalization, sibling stage | 0.9760 | **0.953** |
| v7 | cross-encoder v2 (2× data + French pseudo-labels), band 0.005–0.995 | 0.9786 | pending |

\* "Realistic" validation drops false positives whose true owner is outside
the validation sample (on test every S1 entity competes for its records).
The pipeline's own logged validation F0.5 includes them and reads ~1 point
lower (e.g. v5 LightGBM: 0.9601 logged vs 0.9627 realistic).

The leaderboard sits ~2.3 points under realistic validation every time.
Best explanation: France (15% of test, zero training rows). Solving
`0.85 * val + 0.15 * F_fr = LB` gives F_fr ≈ 0.80.

### Key findings (each measured)

- Ground truth never assigns one S2/S3 record to two S1 entities; ~26% of
  S2/S3 records are unmatched. Enforcing this rule alone: 0.789 → 0.802.
- No leakage: no ID, row-order or ID-number structure within clusters; no
  train/test record overlap.
- `src/normalize.py`'s ASCII-only encode deleted every Indic-script name;
  `normalize_v3` keeps them and adds phonetic consonant skeletons so
  "आनंद हेल्थकेयर" and "Anand Healthcare" both become `nd ltkr`.
- Blocking recall (full train): combined list alone 95.44%; union with
  name-only (+skeletons) and address-only lists 96.34%.
- Error budget on validation (v3): false positives ~1.1 pts, classifier
  misses ~2.1, blocking misses ~1.7; perfect-classifier ceiling ~0.983.
- Cross-encoder (multilingual MiniLM-L12) beats LightGBM on uncertain
  pairs (93% vs 85–89% accuracy); its blend was the biggest single gain.
- Sibling-consistency stage (similarity to the entity's other confident
  matches): +0.2–0.26 on validation.

### Rejected ideas (measured, do not retry blindly)

Expected-F0.5 decoding (no gain); capping matches/entity (cap 5 costs 0.37);
CE v1 + v2 ensemble (v1 weight ≈ 0); wider CE band 0.001–0.999 (no gain);
skeleton tokens in the combined blocking matrix (recall −2.5 pts); stage-2
cross-entity competition features (+0.16 on validation but computed over
only the sampled entities, so untrustworthy on test — `--context`, off).

### Operational lessons (these cost hours)

- Only one heavy job at a time on this 15GB laptop. Test inference with
  107M pairs peaks ~13GB; LightGBM training on 137M candidates ~15GB plus
  swap. Running a second heavy job alongside once froze the machine.
- pandas groupby-rank over 100M+ rows is a memory bomb; `_rank_gap` is a
  numpy lexsort version.
- A row-major memmap written column by column made inference IO-bound for
  an hour; it is column-major now.
- `pgrep -f`/`pkill -f "<pattern>"` also matches the shell running the
  command if the pattern appears in it — it killed our own shell three
  times. Kill by PID.
- Cross-encoder inference: length-sorted batches are ~1.5× faster. Two GPU
  jobs fit in 6GB only with batch 32 for training; a paused (SIGSTOP)
  process keeps its VRAM.

### Files

- Candidates: `v3/cands_{train,test}_multi.npz` (positions into S1 and
  concat(S2,S3) as `read_source_tsv` orders them).
- Models: `model_v5.joblib` (+ `.val.npz` validation pair scores),
  `ce/model` (CE v1), `ce2/model` (CE v2), `ce2/blend_params_v7.json`,
  `ce2/stage2_v7.joblib`.
- Outputs: `output_v5_s2/` (scored 0.953), `output_v7/` (final candidate).
- Logs for every run: `v3/*.log`, `ce/*.log`, `ce2/*.log`.

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

## Follow-up session (same day, 2026-09-26) — leaderboard result, a second
## structural blocking bug, and two more full-scale crashes

This picks up right where the above left off. TL;DR: `model_v2.joblib`'s
output was submitted and scored **0.788**, not the 0.8341 validation number
— a real, ~5.5-point gap, most likely explained by France (see below). While
investigating, a second real blocking bug was found (address-prefix
matching, independent of the France question) and fixing it caused two more
full-scale OOM/thrashing crashes, for reasons distinct from every landmine
already documented above.

### 1. Leaderboard score (0.788) vs. validation score (0.8341) — the France
### hypothesis

`model_v2.joblib`'s already-generated `output/matching_results.tsv` /
`output/candidate_pairs.tsv` (from the original session, previously
unconfirmed-submitted) were submitted this session. Leaderboard score:
**0.788**. That's a meaningful gap from the 0.8341 measured on the held-out
validation split, and validation splits normally track leaderboard scores
fairly closely when the split is representative — so the size of this gap
demanded an explanation, not a shrug.

Leading hypothesis, not yet confirmed with a dedicated diagnostic: **country
distribution**. `CLAUDE.md`'s own binding constraint says train is US/India
only, but test adds **France** as a third country. Counting
`test_source1.tsv`'s country column directly: 663,106 US / 809,986 India /
**259,452 France** (~15% of ~1.73M test entities). The validation split, by
contrast, was carved out of the *train* file and so contains **zero**
France entities — meaning validation F_0.5 was never actually measuring
performance on 15% of what the real test set contains.

Nothing in the pipeline has any France-specific handling: `src/normalize.py`'s
`ABBREVIATION_MAP` is English/Indian-registration vocabulary only
(`ltd`/`pvt`/`llc`/`corp`/etc., no French legal-suffix or address-abbreviation
equivalents), and the sentence-transformer embedder used for blocking
(`all-MiniLM-L6-v2`) is primarily English-trained. If French entities are
blocked/matched substantially worse than US/India as a result, a quick
sanity check on the arithmetic: if US/India together score ~0.85 and France
scores ~X, a 0.85/0.15 macro-ish blend landing at 0.788 implies X ≈ 0.44 —
a plausible magnitude for "this segment gets zero special handling and
partially fails blocking or normalization." This is **a hypothesis**, not
yet verified — the next session should build a France-specific blocking-
recall diagnostic (same method as the India one in section 2 below: sample
France-country S1 entities with known ground-truth matches, measure
per-strategy and union recall) before assuming this is the whole
explanation, or before spending time on a France-specific fix.

### 2. A second real blocking bug, independent of France:
### `address_prefix_candidates` almost never matches

While investigating the France hypothesis, a stratified diagnostic sample
was built: 1500 India-country S1 entities whose ground-truth matches
include at least one **non-Latin-script** S2/S3 business name (e.g. "Red
Ventures Private Limited" ↔ "रेड वेंचर्स प्राइवेट लिमिटेड" — a literal
transliteration). This surfaced a real dataset fact worth remembering:
`train_source2.tsv` is **45% non-Latin-script** business names, `source3`
**37%** (measured directly: `grep -P '[^\x00-\x7F]'` on the India-country
rows of each file). The `normalize_text` function in `src/normalize.py`
does `.encode("ascii", "ignore")`, which **deletes** all Devanagari/Tamil/
etc. script entirely — so both `token_overlap_candidates` and the name-half
of `embedding_knn_candidates`'s input text get an empty/near-empty string
for these rows.

The first fix attempt targeted this directly: swap the embedder for
blocking to a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`,
same 384-dim, drop-in for the existing FAISS/PQ code) and feed it
unicode-preserving text instead of the ascii-stripped version. Direct
similarity checks on known transliterated pairs confirmed the multilingual
model *does* find real signal that the English-only model misses (0.32-0.64
cosine similarity vs. 0.02-0.12). **But it made no measurable difference to
diagnostic recall** (union recall 0.7867 current vs. 0.7869 with the fix)
— because `address_prefix_candidates` was already covering most of these
cases through the (Latin-script) address, and swapping the embedder didn't
change that ceiling.

So the real culprit was `address_prefix_candidates` itself. It matches only
on the literal *first-3-tokens* of `business_address`, treated as one atomic
key. Manually inspecting missed pairs showed why this fails almost
universally on India (and plausibly elsewhere): S2/S3 addresses are
routinely **reordered and abbreviated** relative to S1 — house number moved
to the front, state name contracted to a 2-letter code (`Haryana` → `HR`),
street-level detail dropped entirely. Example real pair: S1 address `"1502,
Tower Olive Omaxe, Badhkal Road, Sector-43, Faridabad, Haryana"` vs. the
matching S2 address `"B3/1502, Faridabad, HR"` — these share zero
first-3-token prefix, but obviously share `"1502"` and `"faridabad"` as
individual tokens.

**Fix:** added `address_token_overlap_candidates` to `src/blocking.py` —
same `_weighted_top_k_candidates` machinery as `token_overlap_candidates`
(doc-frequency-capped, weighted, top-k-per-entity), just applied to *all*
tokens of `business_address` instead of a positional 3-token prefix. Swapped
it in at `src/infer.py`'s `generate_candidates` in place of
`address_prefix_candidates` (which is left in `blocking.py`, still covered
by its existing tests, just no longer wired into the real pipeline).

Measured on the 1500-entity diagnostic sample (5846 true positive pairs):

| Strategy | Recall |
|---|---|
| address-prefix only (old) | 37.2% |
| address-token-overlap only (new) | 89.6% |
| name-token + address-prefix union (old 2-strategy) | 68.5% |
| name-token + address-token-overlap union (new 2-strategy) | 95.6% |
| name-token + address-token-overlap + embedding union (all 3, new) | 96.1% |

This is a much bigger, more structural lever than the France question or
any further top_k tuning — worth remembering as *the* fix from this session
if only one thing survives.

### 3. Two more full-scale crashes — this fix has its own, different memory
### landmine

Given the size of the recall win, two attempts were made to retrain
`model_v3.joblib` with this fix at full scale:

**Attempt 1**: kept the fix, but *also* took the opportunity to restore
`TOP_K_PER_STRATEGY` from 25 back to 50 (and embedding `top_k` from 8 to 20),
reasoning that this exact combination's candidate-dict memory (~12.76GB)
was already measured safe in the original session (see section 3, way
above) once the `del`/`gc.collect()` fixes were in place. **This reasoning
was wrong**: that 12.76GB estimate was measured against the *old*
`address_prefix_candidates` (one key per entity, minimal row count) — it
never accounted for a strategy that explodes each address into many tokens.
Killed after ~2hrs of real run time: swap climbed to 18-20GB with sustained
30-270MB/s swap in/out and a context-switch storm (up to 1.57M/2s vs. a
normal ~10-20k) — genuine thrashing, not a false alarm.

**Attempt 2**: reverted `TOP_K_PER_STRATEGY` back to 25 and embedding
`top_k` back to 8 (the exact config already proven safe for
`token_overlap_candidates` at full scale) — isolating the
`address_token_overlap_candidates` swap as the *only* change from the known-
good 0.8341 run. **Still crashed**, worse this time (iowait up to 66%, up to
10 blocked processes, swap approaching 25GB of the full 31GB) after ~2.5hrs.
This proved the problem isn't `top_k` at all — it's that
`address_token_overlap_candidates` needed its own memory bounds from the
start, not a reuse of `token_overlap_candidates`'s.

**Root cause, once actually reasoned through**: business addresses tokenize
into far more `(entity, token)` rows than business names do (~6-10 tokens
per address after `tokenize(normalize_text(...))`, vs. ~2-3 per name after
stopword filtering). `_weighted_top_k_candidates`'s per-batch merge
(`batch_keys.merge(capped_other, on="key")`) scales with total
`(s1_row, key)` pairs in the batch, not just `batch_size` — so the *same*
`S1_BATCH_SIZE=50_000` / `max_doc_freq=500` bound that keeps
`token_overlap_candidates` safe is proportionally too loose for the
address-token version, by roughly the ratio of average tokens per address to
average tokens per name.

**Fix (applied, not yet fully verified at full scale as of this update)**:
gave `address_token_overlap_candidates` its own, tighter, dedicated
constants — `MAX_ADDRESS_TOKEN_DOC_FREQ = 150` (down from reusing the
500 that `MAX_ADDRESS_PREFIX_DOC_FREQ`/`MAX_TOKEN_DOC_FREQ` use) and
`ADDRESS_TOKEN_BATCH_SIZE = 25_000` (half of `S1_BATCH_SIZE`) — rather than
touching `token_overlap_candidates`'s bounds at all. Both are documented in
`src/blocking.py` with the reasoning above.

Before committing to a third multi-hour `--use-embeddings` run, a
**blocking-only** (no embeddings) full-scale verification script was
written (`token_overlap_candidates` + `address_token_overlap_candidates`
against the real, full 2,206,821-entity `train_source1.tsv` and
10,320,219-row combined others, watching `free -h`/`vmstat` throughout) —
this is the cheap check that should have been run *before* either of the
two crashes above, not after. Check whether that verification actually
passed, and whether a `model_v3.joblib` exists, before re-attempting a full
`--use-embeddings` run.

### 4. Other notes from this session

- `EMBED_BATCH_SIZE` (the SentenceTransformer encode batch size, GPU-VRAM-
  bound, unrelated to the RAM/swap issues above) was raised 256 → 2048.
  Checked live against a running encode job: 787MiB/6141MiB VRAM in use at
  batch_size=256 on this RTX 4050, i.e. ~5.3GB of genuine headroom. This is
  a real GPU-VRAM lever, distinct from every RAM/swap constant discussed
  above — don't conflate the two categories of "memory" when reasoning about
  safe values here. Unverified on a real full-scale GPU run as of this
  update (both attempted retrains crashed during CPU-bound phases before or
  after the GPU phase, not during it, so this change was never actually
  exercised end-to-end at full scale).
- Running two memory-heavy full-scale-ish Python processes at once on this
  15GB machine (a verification script plus a small-sample
  `--candidates-cache` round-trip test, launched back-to-back without
  waiting for the first to finish) transiently dropped available RAM to
  ~80MB. Neither process had a bug — they were just competing for the same
  limited RAM simultaneously. Don't parallelize memory-heavy checks on this
  machine; run them sequentially even when each one individually looks
  cheap.
- `--candidates-cache`'s round-trip is **still unverified** as of this
  update (the attempt this session was accidentally killed by the point
  above before it could write the cache file) — this was already flagged as
  unverified in the original session (see item 5 in "If you're picking this
  up next," above) and remains so.
