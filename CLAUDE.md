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

## Status (as of 2026-09-25)

- Design spec and implementation plan are written and committed — see "Where
  to look" below. Both were revised once already after the real dataset's
  scale (millions of rows/file) was discovered to break the original
  dense-matrix/iterrows-based design.
- Real dataset downloaded and extracted to `student_resource/dataset/`
  (2.2M–5.3M rows per source file, ~2.5GB total, gitignored).
- **No pipeline code written yet** — `code/business_entity_resolution/`
  does not exist. Execution (subagent-driven-development against the plan)
  has not started.
- **Training has not been run.** No model artifact exists.
- What's left: implement all 11 plan tasks (I/O utils, normalize, blocking,
  features, metrics, train, infer), run the pipeline against the real
  dataset, tune the classification threshold, fill in
  `student_resource/Documentation_template.md`, assemble the final
  submission zip.

## Where to look

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

## Commands (once code exists — see plan Task 1/11 for exact versions)

```bash
pip install -r code/business_entity_resolution/requirements.txt

# Run the full test suite
cd code/business_entity_resolution && python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_blocking.py::test_token_overlap_candidates_matches_shared_significant_token -v

# Train (writes model.joblib)
python -m src.train_cli --dataset-dir student_resource/dataset/train --model-path model.joblib

# Infer (writes output/matching_results.tsv and output/candidate_pairs.tsv)
python -m src.infer_cli --dataset-dir student_resource/dataset/test --model-path model.joblib --output-dir output

# Validate output format before submitting
python3 student_resource/utils/validate_submission.py \
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
