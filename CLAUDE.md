# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Build an entity-resolution pipeline (codename **Kinfolk**) for the Amazon ML
Challenge 2026 "Business Entity Resolution Challenge": match noisy business
records across 3 sources (Source 1 is the deduplicated reference; find all
matching Source 2 / Source 3 records for each Source 1 entity), scored by
macro-averaged F_0.5.

## Deadline

Challenge window: 25 Sep 2026 00:00 IST – 27 Sep 2026 23:59 IST. Max 5
leaderboard submissions/day. The public leaderboard scores a subset of test;
final ranking uses the private remainder. All teams submit a final zip
(`output/` + `code/business_entity_resolution/` + methodology document);
top-100 packages are reviewed.

## Status (as of 2026-09-27 afternoon)

See `HANDOFF.md` for the narrative of both sessions. Current state:

- **The active pipeline is "v3"** (`src/*_v3.py`, `src/cross_encoder.py`,
  `src/blend_cli.py`, `src/stage2_cli.py`). The original plan-era modules
  (`src/blocking.py`, `src/features.py`, `src/train.py`, `src/infer.py`,
  `*_cli.py`) are only partly used now: `infer.assign_matches`, `metrics`,
  `utils_io` and `train.split_train_validation` are shared; the rest is the
  superseded v2 pipeline.
- **Best leaderboard score: 0.953** (27 Sep 12:41, `output_v5_s2/`).
  Submission history: 0.789 (v2) → 0.802 (v2 + one-S1-per-record rule) →
  0.930 (v3) → 0.934 (v3b) → 0.943 (v3b + cross-encoder) → **0.953**
  (v5 + cross-encoder + stage 2). A teammate's separate file scored 0.773.
- **v7** (v5 LightGBM + cross-encoder v2 with French pseudo-labels, wider
  uncertainty band, stage 2) is the final candidate: realistic validation
  F0.5 **0.9786** vs 0.9760 for the 0.953 submission. Output dir
  `output_v7/`. Check `HANDOFF.md` for whether it was submitted and scored.
- **Validation vs leaderboard gap is ~2.3 points every time**, most likely
  France (15% of test, absent from train). Back-of-envelope estimate puts
  French F0.5 around 0.80 vs ~0.97 for US/India. Unmeasurable without
  labels.
- **Measured ceilings:** blocking recall 96.34% of true train pairs
  (US 98.0%, India 93.8%); a perfect classifier on our candidates would score
  ~0.983 on validation.
- **Ideas tested and rejected (do not retry without new evidence):**
  expected-F0.5 per-entity decoding (no gain vs a fixed threshold); capping
  matches per entity (cap 5: 0.9622 → 0.9585); ensembling cross-encoder v1
  with v2 (v1 weight ≈ 0); widening the cross-encoder band beyond
  0.005–0.995 (no gain); skeleton tokens inside the combined blocking matrix
  (recall 95.5% → 93.0%); stage-2 cross-entity "competition" context
  features (the +0.16 validation gain is not trustworthy because validation
  only has the sampled entities competing; code is in `stage2_cli.py` behind
  `--context`, off by default).
- 92 tests pass (`pytest tests/`).
- Methodology document: `student_resource/Documentation.md` (filled in).

## Where to look

- `HANDOFF.md` — narrative of the 2026-09-26 session (v2, crashes) and the
  2026-09-26/27 session (v3 → v7, every measurement behind the design).
- `student_resource/Documentation.md` — methodology write-up for the final
  package (numbers, features, reproduction commands).
- `docs/superpowers/` — the original design spec / plan for the v2 pipeline.
  Historical; the v3 modules supersede its blocking/feature design.
- `student_resource/utils/validate_submission.py` — official format
  validator; run before every leaderboard submission (`--check-ids` too).
- Challenge PDFs in the repo root — rules and the exact metric.

## Architecture (v3, current)

```
raw TSVs (2.2M S1 / 10.3M S2+S3 train; 1.73M / 9.97M test)
  -> normalize_v3: fold accents but keep Indic scripts, strip leading zeros,
     expand US/Indian/French address abbreviations, US states -> codes,
     French départements/régions + Indian state names/codes -> one region
     token, split legal forms from the name core, phonetic consonant skeletons
  -> blocking_v3 (pipeline_v3 block --multi): hashed-token CSR matrices,
     per-country cosine top-k; union of combined top-40, name+skeleton
     top-20, address top-20 -> ~62 candidates/entity, 96.34% recall
  -> features_v3: rapidfuzz cpdist string features + global context
     (rank/gap within entity and among S1 entities competing for a record),
     written to a column-major memmap at inference
  -> LightGBM (pipeline_v3 train/infer), early stopping, 200k train entities
  -> cross_encoder: multilingual MiniLM-L12 (Apache-2.0) fine-tuned on hard
     pairs; blend_cli re-scores pairs with LightGBM prob in [lo, hi] via a
     logistic blend of logits fitted on validation
  -> stage2_cli: sibling-consistency features + small LightGBM
  -> infer.assign_matches: each S2/S3 record kept only for its best S1
     entity, then threshold -> matching_results.tsv
```

**Binding scale constraint:** never build dense matrices across full
sources and never loop over full sources in Python per pair; everything is
sparse matrices, numpy, or rapidfuzz `cpdist` over aligned lists.

**Memory constraint (15GB RAM, 6GB VRAM laptop):** only one heavy CPU job at
a time (blocking, LightGBM train, inference). Two at once froze the laptop
once and thrashed swap twice. Watch `vmstat` swap-in (`si`): sustained
>50MB/s means kill the job. `pgrep -f`/`pkill -f` with a pattern that also
appears in your own shell command line kills that shell — match by PID.

## Commands

Use `/home/vinay/amzon-ml/.venv/bin/python`. Run pipeline modules from
`code/business_entity_resolution/`.

```bash
# tests
python -m pytest tests/ -q

# blocking (~3h train, ~2.5h test on this laptop)
python -m src.pipeline_v3 block --multi --split train --dataset-dir ../../student_resource/dataset/train --out ../../v3/cands_train_multi.npz
python -m src.pipeline_v3 block --multi --split test  --dataset-dir ../../student_resource/dataset/test  --out ../../v3/cands_test_multi.npz

# LightGBM (~20 min) and test inference (~1h incl. writing scores.tsv.gz)
python -m src.pipeline_v3 train --dataset-dir DATA/train --candidates cands_train_multi.npz --model-path model_v5.joblib
python -m src.pipeline_v3 infer --dataset-dir DATA/test --candidates cands_test_multi.npz --model-path model_v5.joblib --output-dir output_v5

# cross-encoder (prepare ~3 min, train ~1h per 900k pairs on the laptop GPU)
python -m src.cross_encoder prepare --dataset-dir DATA/train --candidates cands_train_multi.npz --out-dir ce2
python -m src.cross_encoder train --data-dir ce2/data --model-dir ce2/model --batch-size 32 --lr 2e-5

# blend + stage 2 (fit on validation, apply to test)
python -m src.blend_cli fit   --dataset-dir DATA/train --val-scores model_v5.joblib.val.npz --model-dir ce2/model --params-out blend.json --lo 0.005 --hi 0.995
python -m src.blend_cli apply --dataset-dir DATA/test --scores-dir output_v5 --model-dir ce2/model --params blend.json --output-dir output_v7_ce
python -m src.stage2_cli fit   --dataset-dir DATA/train --val-scores model_v5.joblib.val.npz --model-dir ce2/model --blend-params blend.json --model-out stage2.joblib
python -m src.stage2_cli apply --dataset-dir DATA/test --scores-dir output_v5 --ce-dir output_v7_ce --stage2-model stage2.joblib --output-dir output_v7

# validate before every submission
python student_resource/utils/validate_submission.py --matching output_v7/matching_results.tsv --candidate output_v7/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

## Key constraints (binding, from the challenge rules)

- `country` is an open string set (train: US/India; test adds France) —
  never filter or one-hot to a fixed list. (The region-name map in
  `normalize_v3` only rewrites address tokens; it does not restrict
  countries.)
- No external data/API lookups — disqualifying. Pretrained model weights
  are fine; no internet augmentation of records.
- Final matching models must be MIT/Apache-2.0 and ≤8B parameters
  (LightGBM MIT; MiniLM-L12 Apache-2.0, ~118M).
- Output TSVs: tab-separated, comma-separated ID lists, no quoting, no
  duplicates, one row per Source 1 entity; matches ⊆ candidates.
- Ground truth never assigns one S2/S3 record to two S1 entities — keep
  `assign_matches`' one-S1-per-record rule.
