# Business Entity Resolution — Design Spec

Date: 2026-09-25
Challenge: Amazon ML Challenge 2026, "Business Entity Resolution Challenge"

## Problem

Match business records across 3 noisy data sources. Source 1 is the deduplicated
reference; for each Source 1 entity, find all matching records in Source 2 and
Source 3 (zero, one, or many). Data is TSV. No shared IDs; matching relies on
noisy `business_name`, `business_address`, and `country` fields.

Full requirements, output format, and scoring are in the two PDFs at the repo
root:
- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`
- `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`

## Constraints (from problem statement)

- Output: `output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv`
  (audit only, must be a superset of the matches — the exact candidate set fed
  to the final model, post-blocking).
- `country` is an open string set: train has US/India, test adds France. Must
  not hardcode or filter to a fixed country list anywhere in the pipeline.
- No external data/API lookups (geocoding, business registries, etc.) —
  disqualifying if detected.
- Final model must be MIT/Apache-2.0 licensed and ≤8B parameters.
- Scoring: F_0.5 (β=0.5), precision-heavy, macro-averaged per Source-1 entity.
  Singletons (correctly predicted empty match list) score 1.0; a false merge on
  a singleton scores 0.0.
- `utils/validate_submission.py` (to be supplied by the user from the
  challenge portal) checks output format before submission.

## Environment

- GPU: NVIDIA RTX 4050 (~6GB VRAM). Enough for a small sentence-transformer
  bi-encoder (e.g. `all-MiniLM-L6-v2`, ~90MB) used only for blocking/feature
  embeddings — not for an LLM-scale final model.
- Final matching-stage model: LightGBM (or XGBoost) binary classifier over
  engineered similarity features, chosen over an embedding+cross-encoder or
  LLM approach. Rationale: precision-heavy F_0.5 favors a calibrated,
  threshold-tunable tabular classifier; trivially MIT-licensed; no param-count
  risk; fast to iterate without heavy GPU dependence at inference time.
- RAM: 15GB total, ~9.5GB available. 12 CPU cores.

## Scale (real dataset, measured after download)

- `train_source1.tsv`: 2,206,822 rows. `train_source2.tsv`: 5,034,617 rows.
  `train_source3.tsv`: 5,285,604 rows. `train_ground_truth.tsv`: 2,206,822 rows
  (one row per Source-1 training entity, matching `matching_results.tsv`'s own
  "every entity gets exactly one row" rule).
- `test_source1.tsv`: 1,732,545 rows. `test_source2.tsv`: 4,887,274 rows.
  `test_source3.tsv`: 5,082,317 rows.
- Raw file sizes range 127MB–509MB each; total dataset ~2.5GB uncompressed.

This rules out anything with O(n×m) memory or time cost across full source
pairs (a dense S1×S2 similarity matrix would be ~2.2M × 5M = 11 trillion
cells) and anything using row-at-a-time Python iteration (`DataFrame.iterrows()`)
over millions of rows. Every stage below is written as vectorized
pandas/numpy operations, SQL-style joins, or an approximate-nearest-neighbor
index — never a Python loop over the full row count of a source file.

## Pipeline

```
dataset (raw TSVs)
  -> normalize (shared text cleaning, both name + address, all sources)
  -> block (multi-key candidate generation -> candidate_pairs.tsv)
  -> feature engineer (per S1/candidate pair)
  -> GBM classify (probability per pair)
  -> threshold + per-entity assignment -> matching_results.tsv
  -> validate_submission.py
```

### 1. Normalization

Shared module (`normalize.py`) applied identically to `business_name` and
`business_address` across all 3 sources, train and test:
- Lowercase, strip punctuation, collapse whitespace
- Expand common legal-suffix / address abbreviations via a suffix-map table
  (corp/corporation, pvt/private, ltd/limited, rd/road, st/street, etc.) —
  the map is a data table, not per-country hardcoded logic
- Tokenize for downstream token-based features
- `country` field is kept as a raw string; used only as a grouping/feature
  signal, never filtered against a fixed set

### 2. Blocking (candidate generation)

Union of three strategies to maximize recall (recall ceiling is set here per
the problem statement's own tip). All three are implemented as vectorized
joins or an ANN index — never a per-row Python loop over a full source file.

1. **Token-overlap join.** Build a long-format `(entity_id, token)` table per
   source (normalize + tokenize the `business_name` column, `explode()` the
   token lists, drop stopwords). Inner-join the S1 token table against the
   S2/S3 token table on `token`, then `groupby(entity_id_s1)` to collect the
   matched other-side ids into a set. This is a single pandas merge + groupby
   (C-level), not a Python loop.
   **Block-size cap:** before joining, drop any token whose document
   frequency (row count in the exploded table) exceeds a cap (default: top
   0.1% most frequent tokens, or an absolute cap of a few thousand records,
   whichever is smaller). Without this cap, a common token shared by
   hundreds of thousands of records would blow up the join into an
   unusably large (and mostly useless) candidate set. Capped-out tokens
   contribute no candidates via this strategy — the other two strategies
   still cover those entities.
2. **Address-prefix join.** Same join pattern, but the per-entity key is a
   single string — the first N normalized address tokens joined together —
   instead of an exploded token list. Records with fewer than N tokens key
   on whatever they have; records with zero address tokens simply produce no
   key and get no candidates from this strategy (never an error).
3. **Embedding kNN via FAISS.** Bi-encoder (`all-MiniLM-L6-v2`) embeds
   concatenated normalized name+address text, batched on GPU. Candidates are
   found with a FAISS approximate-nearest-neighbor index
   (`IndexIVFFlat`, falling back to `IndexFlatIP` for partitions too small to
   train an IVF index), never a dense S1×other similarity matrix. **To keep
   both the FAISS index and the embedding matrix within the ~9.5GB available
   RAM / 6GB VRAM budget, this strategy partitions by `country` string** —
   an index is built per distinct country value seen in the other-source
   partition, and each S1 entity only queries its own country's index. This
   is a compute/memory trade-off, not the "hard filter" the constraints
   section prohibits for the *pipeline's final candidate set* — cross-country
   candidates are still reachable through the token-overlap and
   address-prefix strategies (strategies 1–2 are never country-partitioned),
   so a true match with a mismatched or missing country label is not
   silently unreachable, only unreachable via this one of three strategies.

`country` string match is otherwise a soft boost (used as a classifier
feature, see below), never a hard filter on the final unioned candidate set.

The union of all three strategies' output **is** `candidate_pairs.tsv` — the
exact set fed to the classifier, per the problem statement's requirement that
this file reflect the last stage before model scoring, not raw blocking
output.

### 3. Feature engineering

Per (S1 entity, candidate) pair, computed over the candidate-pair table
produced by blocking (millions of pairs, not full source cross products):
- **Name**: Levenshtein ratio, token Jaccard, common-token count,
  abbreviation-normalized exact-match flag
- **Address**: same battery + numeric-token overlap (PIN/zip/house number) +
  token-order-invariant similarity
- **Embedding cosine similarity**: reuses the embedding vectors already
  computed during blocking (indexed by entity_id) — a row-wise dot product
  over aligned index arrays, not a fresh embed-per-pair call
- **Country**: exact-match flag (feature, not filter)
- **Structural**: name length diff, token count diff

**TF-IDF cosine descoped.** The original design listed a TF-IDF cosine
feature; it's dropped to keep the implementation bounded — Levenshtein +
token Jaccard + embedding cosine already give three complementary signals
(edit distance, exact token overlap, semantic similarity), and a global
TF-IDF fit across 10M+ records is added implementation surface for
likely-marginal additional signal. Revisit only if validation F_0.5 shows a
clear gap these three don't cover.

**Vectorized computation, not per-pair Python calls.** With blocking
producing on the order of tens of millions of candidate pairs, calling a
Python function once per pair inside a `for _, row in df.iterrows()` loop is
not viable — `iterrows()` reconstructs a pandas Series per row, which is
especially slow. Feature-building loops use `itertuples()` (no Series
reconstruction, several times faster) at minimum, and batch where the
underlying library supports it (e.g. `rapidfuzz.process.cdist` for computing
many string-pair ratios in one call instead of one Python-level `fuzz.ratio()`
call per iteration). Country-match, structural (length/token-count diffs),
and numeric-overlap features are computed with vectorized pandas/numpy
string and arithmetic operations over the whole candidate-pair DataFrame at
once where practical.

### 4. Model

LightGBM binary classifier (MIT-licensed, no param-count concern since it's
not an LLM). Training pairs:
- Positives: from `train_ground_truth.tsv`
- Negatives: hard negatives — non-matching pairs that still passed blocking
  (same-key false candidates), plus some random negatives for coverage
- **Negative sampling cap.** With 2.2M Source-1 training entities and
  blocking potentially returning dozens of candidates each, the raw
  candidate-derived negative pool can run into the tens of millions of rows
  — enough to make feature computation and LightGBM training slow without
  improving the model (most negatives are easy/uninformative once a
  representative sample is present). Cap negatives per Source-1 entity to a
  fixed multiple of that entity's positive count (default 10:1 negative:positive,
  sampled uniformly at random from that entity's blocking candidates when
  the pool exceeds the cap), rather than including every candidate as a
  labeled row.

### 5. Training / validation

- Hold out a slice of Source-1 training entities (and their true S2/S3
  matches) as a validation split — never touched during feature/model fitting.
- Compute F_0.5 macro-averaged per Source-1 entity exactly per the PDF
  formula, on the validation split.
- Tune the classification threshold on validation to maximize F_0.5 (expect a
  high threshold given the precision weighting).
- Per-S1-entity output: keep all candidates scoring above threshold; empty
  list if none clear the bar. No entity is dropped from output (every test
  Source-1 entity gets exactly one row, per format rules).

### 6. Repo structure

```
code/business_entity_resolution/
  src/
    normalize.py
    blocking.py
    features.py
    train.py
    infer.py
    utils_io.py      # tsv read/write matching the PDF's exact schema
  README.md           # exact reproduce steps: data -> blocking -> matching -> output
  requirements.txt    # pinned deps
output/
  matching_results.tsv
  candidate_pairs.tsv
Documentation_template.md   # filled in after model is built
```

Run order: `train.py` (fit + save model artifact) → `infer.py` (blocking +
features + scoring + writes both TSVs) → `validate_submission.py`.

**I/O at scale.** Source TSVs are 127MB–509MB each. Read with
`pd.read_csv(..., sep="\t", dtype=str)` per file, one file resident at a
time where possible rather than holding all six train+test source files in
memory simultaneously — train.py only needs the train files, infer.py only
the test files, and within each, S2/S3 are only combined (`pd.concat`) at
the point blocking needs a unified other-source frame, not earlier.

### 7. Validation gate

`infer.py` ends by invoking the user-supplied
`utils/validate_submission.py` automatically, so format errors are caught
locally before spending a leaderboard submission.

## Resolved items

- `dataset/train/*.tsv`, `dataset/test/*.tsv`, `utils/validate_submission.py`,
  and `Documentation_template.md` are now present in the repo, under
  `student_resource/` (unzipped from `6ab10eb3b23ba_student_resource.zip`).
  Real sizes are documented in the Scale section above.
  `train.py`/`infer.py` still fail fast with a clear error if an expected
  file is missing — real data doesn't remove that requirement, it's still
  the correct behavior if a path is wrong or a file gets moved.

## Out of scope for this spec

- Hyperparameter search automation (manual/default LightGBM params to start)
- Any UI/dashboard
- Cross-encoder reranking stage (deferred; GBM-only approach chosen)
- Distributed/multi-machine processing (single-machine vectorized pandas +
  FAISS is the target; if that proves insufficient at real data volumes
  during implementation, that's a fix-loop finding, not a re-scope)
