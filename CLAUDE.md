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
leaderboard submissions/day. Top-100 teams must later submit methodology +
source code + candidate-generation/blocking strategy details.

## Status (as of 2026-09-26)

See `HANDOFF.md` for a full narrative of this session (what broke, why, and
how it was fixed) — this section is just the current-state summary.

- Design spec and implementation plan are written and committed — see "Where
  to look" below. Both were revised once already after the real dataset's
  scale (millions of rows/file) was discovered to break the original
  dense-matrix/iterrows-based design.
- Real dataset downloaded and extracted to `student_resource/dataset/`
  (2.2M–5.3M rows per source file, ~2.5GB total, gitignored).
- **Pipeline code is implemented** at `code/business_entity_resolution/`
  (all 11 plan tasks). 79 tests pass (`pytest tests/ -v`).
- **Training has been run twice, end-to-end, on the real full-scale
  dataset:**
  - `model.joblib` — first successful run, token+address blocking only (no
    embeddings). Validation F_0.5 = **0.6302**, threshold 0.75. Kept as a
    known-good fallback artifact; do not overwrite.
  - `model_v2.joblib` — second run, with `--use-embeddings` (semantic
    blocking added on top of token+address). Validation F_0.5 = **0.8341**,
    threshold 0.70, 33,212,774 training pairs, 441,364 validation entities.
    This is the current best model.
- **Inference has been run on the real test set** with `model_v2.joblib`:
  `output/matching_results.tsv` and `output/candidate_pairs.tsv` exist and
  passed `validate_submission.py` (PASS, no blocking issues). Not yet
  confirmed submitted to the leaderboard — check with whoever picks this up
  before assuming it was.
- **Machine is memory-constrained (15GB RAM, RTX 4050 laptop GPU, 6GB
  VRAM).** Getting `model_v2.joblib` trained took 4 failed/killed attempts
  before succeeding — see `HANDOFF.md` and the comments in `src/blocking.py`
  / `src/train.py` / `src/features.py` for the specific fixes. Do not change
  `TOP_K_PER_STRATEGY`, `MAX_TOKEN_DOC_FREQ`, `MAX_ADDRESS_PREFIX_DOC_FREQ`,
  or the embedding `top_k` defaults without re-reading those comments first
  — several of these look like easy wins for recall but caused real OOM/swap
  crises at full (2.2M-entity) scale.
- What's left: decide whether to iterate further (blocking recall is still
  ~79.5% on a sample diagnostic, so there's real headroom above 0.8341),
  fill in `student_resource/Documentation_template.md`, assemble the final
  submission zip, confirm leaderboard submission.

## Where to look

- `HANDOFF.md` — narrative account of the 2026-09-26 session: how the
  pipeline went from "no code" to a validated 0.8341 model, every dead end
  hit along the way (candidate blowup, OOM kills, thrashing), and what a
  fresh pair of eyes needs to know before touching `src/blocking.py` or
  `src/train.py` again.
- `docs/superpowers/specs/2026-09-25-business-entity-resolution-design.md` —
  design spec, authoritative for **why** (constraints, scale numbers,
  architecture rationale).
- `docs/superpowers/plans/2026-09-25-business-entity-resolution-plan.md` —
  task-by-task implementation plan with exact code for each module,
  authoritative for **how**. Follow this over any other description if they
  disagree.
- `student_resource/README.md`, `student_resource/dataset/` — the
  challenge's own starter kit and the real data.
- `student_resource/utils/validate_submission.py` — the official format
  validator; run before every leaderboard submission.
- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`,
  `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`
  — original challenge rules.
- Root `README.md` predates the scale-driven design revision — it still
  lists a TF-IDF cosine feature that the spec later deliberately descoped.
  Treat the plan doc as authoritative over `README.md` on any conflict.

## Architecture (planned — see the plan doc for exact code per module)

```
dataset (raw TSVs, millions of rows/file)
  -> normalize (src/normalize.py: lowercase, strip punctuation, expand
     legal-suffix/address abbreviations, tokenize)
  -> block (src/blocking.py: union of 3 strategies -> candidate_pairs.tsv)
       1. token-overlap join on normalized business_name tokens, with a
          document-frequency cap so a common token can't blow up the join
       2. address-prefix join on normalized business_address tokens
       3. embedding kNN via a country-partitioned FAISS index
          (all-MiniLM-L6-v2 embeddings; never a dense S1xS2/S3 matrix)
  -> feature engineer (src/features.py: per-pair Levenshtein ratio, token
     Jaccard, common-token count, numeric-token overlap, embedding cosine,
     country-match flag, structural diffs)
  -> LightGBM binary classifier (src/train.py), MIT-licensed, trained on
     ground-truth positives + a capped sample of blocking-candidate
     negatives
  -> threshold tuned on a held-out validation split to maximize F_0.5
     (src/metrics.py implements the exact per-entity F_0.5 macro-average)
  -> per-entity assignment + TSV output (src/infer.py) -> matching_results.tsv
     + candidate_pairs.tsv -> validate_submission.py
```

**Binding scale constraint:** dataset files are 2.2M–5.3M rows each. All
pipeline code must use vectorized pandas joins/groupbys or a FAISS ANN
index — never a dense similarity matrix across full sources, and never a
`DataFrame.iterrows()` loop over a full source file. This is enforced in
the plan's Step 3 code for every blocking/feature/scoring task, not by unit
tests (which use small fixtures and would pass either way).

## Commands

The venv at `.venv/` (repo root) already has all deps installed, including
`torch`+CUDA, `sentence-transformers`, and `faiss-cpu`. Use
`.venv/bin/python`, not a bare `python3`.

```bash
# Run the full test suite (fast, ~3s, uses tiny fixtures)
cd code/business_entity_resolution && /home/vinay/amzon-ml/.venv/bin/python -m pytest tests/ -v

# Run a single test
.venv/bin/python -m pytest tests/test_blocking.py::test_token_overlap_candidates_matches_shared_significant_token -v

# Train on the real full dataset. Takes ~2 hours with --use-embeddings
# (dominated by GPU-encoding the ~10.3M "other" rows once), ~50 min without.
# Run in the background (nohup ... &) and monitor `free -h` / the log file --
# see HANDOFF.md before doing this, the naive version OOMs.
.venv/bin/python -m src.train_cli \
    --dataset-dir student_resource/dataset/train \
    --model-path model_v2.joblib \
    --use-embeddings

# To iterate on classifier/feature/negative-sampling changes without paying
# the ~2hr blocking cost every time, add --candidates-cache <path>: first
# run populates it, later runs against the same path reuse it (untested as
# of 2026-09-26 -- see HANDOFF.md item 5 before relying on it).
#   --candidates-cache candidates_v1.pkl --max-negatives-per-positive 10

# Infer (writes output/matching_results.tsv and output/candidate_pairs.tsv).
# Also ~2 hours with --use-embeddings; pass it to match how the model was
# trained, or blocking recall at inference time won't match validation.
.venv/bin/python -m src.infer_cli \
    --dataset-dir student_resource/dataset/test \
    --model-path model_v2.joblib \
    --output-dir output \
    --use-embeddings

# Validate output format before submitting
.venv/bin/python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

## Key constraints (binding, from the challenge rules)

- `country` is an open string set (train: US/India; test adds France) —
  never hardcode, filter, or one-hot to a fixed list anywhere in the
  pipeline.
- No external data/API lookups (geocoding, business registries, internet
  augmentation) — disqualifying if detected.
- Final matching-stage model must be MIT/Apache-2.0 licensed and ≤8B
  parameters (LightGBM satisfies this trivially; embeddings are blocking-
  stage only, not the final model).
- Output TSVs are tab-separated; ID lists are comma-separated, no quoting,
  no duplicates. Every Source-1 test entity gets exactly one row (empty
  `matched_entity_ids` allowed for singletons).
- `matched_entity_ids` for an entity must be a subset of that entity's
  `candidate_entity_ids` in `candidate_pairs.tsv`.
