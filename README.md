# Amazon ML Challenge 2026 — Business Entity Resolution

Solution pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

## 📌 Problem Overview
In commercial platforms, business identity data originates from multiple independent data sources without shared identifiers. This project resolves records across 3 independent sources:
- **Source 1 (`S1`):** Deduplicated reference source.
- **Source 2 (`S2`) & Source 3 (`S3`):** Noisy secondary sources.
- **Objective:** For every Source 1 entity, find all matching entities in Source 2 and Source 3 (zero, one, or many).

## 📊 Evaluation Metric
Submissions are evaluated using **Macro-averaged $F_{0.5}$ score** across all Source 1 entities:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Precision-Weighted:** Penalizes false merges $2\times$ harder than missed matches.
- **Singletons Scored:** Correctly predicting no matches yields $1.0$; a false match yields $0.0$.

## 🏗️ Architecture & Pipeline
1. **Text Preprocessing & Normalization:**
   - Legal suffix expansion (*Corp*, *Pvt*, *Ltd*), road/street normalization (*Rd*, *St*), punctuation removal, lowercasing.
2. **Multi-Strategy Blocking (Candidate Generation):**
   - Inverted token-index joins with frequency cutoffs.
   - Address prefix key matching.
   - Outputs candidates to `output/candidate_pairs.tsv`.
3. **Feature Engineering:**
   - Rapidfuzz string similarity (Levenshtein, token sort, Jaccard).
   - Numeric token & postal code overlap.
   - Global TF-IDF sparse cosine similarity.
   - Country match flag (open-string handling including unseen test regions like France).
4. **Classification & Thresholding:**
   - Fast LightGBM binary classifier trained on positive matches and hard negative candidates.
   - $F_{0.5}$-optimized decision threshold.
5. **Output Generation:**
   - Formats final matches into `output/matching_results.tsv`.
   - Verified via `student_resource/utils/validate_submission.py`.

## 📁 Repository Structure
```
.
├── README.md
├── 6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf
├── 6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf
├── docs/
│   └── superpowers/
│       ├── specs/2026-09-25-business-entity-resolution-design.md
│       └── plans/2026-09-25-business-entity-resolution-plan.md
└── student_resource/
    ├── README.md
    ├── Documentation_template.md
    ├── utils/validate_submission.py
    └── dataset/           # train/ and test/ TSVs (ignored in git)
```

## 🚀 Usage & Validation
```bash
# Validate output format prior to submission
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
