# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Alt F4
**Team Members:** Kishore P, Reshma Banu, Fahad Ali Khan, Vinay Gk
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We match each Source 1 business to its Source 2 / Source 3 records with a four-stage pipeline:
sparse multi-strategy blocking (TF-IDF cosine over hashed name/address tokens, plus a phonetic
"consonant skeleton" index that lets Indic-script names meet their English originals), a LightGBM
pair classifier over vectorized string and context features, a fine-tuned multilingual MiniLM
cross-encoder that re-scores the pairs the classifier is unsure about, and a sibling-consistency
second stage. A one-Source-1-entity-per-record assignment rule, taken directly from the structure
of the ground truth, removes a whole class of false merges.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the 2.2M-entity training set that shaped the design:

- **Each Source 2 / Source 3 record belongs to at most one Source 1 entity.** Across 7,638,365
  matched records in the training ground truth, none is shared. About 26% of Source 2 / Source 3
  records match nothing (noise / hard negatives).
- **Cluster sizes vary from 0 to 11 matches per entity** (mean about 3.5; 5.6% singletons), with the
  Source 2 and Source 3 counts roughly independent. No per-entity cap is valid: capping at 5 matches
  lowered validation F0.5 from 0.9622 to 0.9585.
- **Name noise:** typos and character swaps ("Coasta1 Aluminum"), reordering, legal-form changes
  (Pvt Ltd / Private Limited, LLC / L.L.C., SARL / S.A.R.L.), domain-name forms
  ("SHARONREGIONALASSOCIATION.COM"), bracket/punctuation junk, and invented brand names over a
  copied address ("D0vakor", "Korvantage").
- **Script changes:** Indian Source 2 / Source 3 names are often in Devanagari, Gujarati, Tamil,
  Telugu, Kannada, Malayalam, Bengali or Odia, and state names appear in native script
  ("महाराष्ट्र"), in English, or as codes ("MH").
- **Address noise:** component reordering, abbreviations (Rd/Road, R./Rue, All./Allée), leading
  zeros on house numbers ("03415" vs "3415"), dropped street detail, US state names vs codes, and
  French départements in place of régions ("Gironde" vs "Nouvelle-Aquitaine").
- **Hard negatives:** records copying an entity's street with a changed house number, or its name
  with one word swapped.
- **France** appears only in the test set (about 15% of test entities), so nothing can be learned
  from French labels directly.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted classifier + cross-encoder re-scoring + second-stage
consistency model (hybrid).
**Core Innovation:** script-independent phonetic blocking and features for multilingual names;
full-candidate-set context features (how a pair ranks among the Source 1 entities competing for the
same record); a cross-encoder applied only where the classifier is uncertain; and a
sibling-consistency stage that checks each candidate against the entity's other confident matches.

---

## 3. Candidate Generation (Blocking)

- **Normalization** (`src/normalize_v3.py`): Unicode folding that keeps Indic scripts intact, dotted
  acronym collapsing (S.A.R.L. → sarl), leading-zero stripping on numbers, US / Indian / French
  address abbreviation expansion, US state names → codes, French départements and régions and
  Indian state names (English, codes, native script) → one canonical region token, and separation
  of legal-form words from the distinctive name core.
- **Blocking keys used:** three sparse TF-IDF (cosine) retrieval lists per country, over hashed
  tokens in CSR matrices (`src/blocking_v3.py`):
  1. combined name-core + address tokens, top-40;
  2. name-only tokens plus phonetic consonant skeletons (Indic letters mapped through their Unicode
     character names, Latin words vowel-stripped with similar consonants folded), top-20;
  3. address-only tokens, top-20.
  Tokens with document frequency above 50,000 are dropped. Search runs in forked workers sharing
  the other-side matrix copy-on-write.
- **Candidate pairs generated:** 107,183,216 for the test set (about 62 per Source 1 entity);
  137,706,058 for training.
- **How we ensured true matches were not lost:** recall was measured against the ground truth at
  every change. Full-train recall: 95.44% with the combined list alone, **96.34%** with the
  three-list union (US 98.0%, India 93.8%). The name-only list with phonetic skeletons recovers
  Indic-script names whose address is too short to retrieve them.

---

## 4. Matching Model

**Features used** (`src/features_v3.py`, computed with `rapidfuzz.process.cpdist`, vectorized):
- **Name features:** ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the name core;
  ratio / partial ratio on the space-free name (domain-form names); legal-form similarity;
  phonetic-skeleton ratio and token-set; name-of-one-side inside the address of the other.
- **Address features:** ratio, token-set, token-sort, partial ratio; street/city-words-only
  token-set; house-number match and ratio; all-numbers equality, ratio and partial ratio; postal
  code match; last-token (region) match; token counts.
- **Context features** over the full candidate set: rank and gap-to-best of the blocking score and
  of name / address similarity within the Source 1 entity's candidates and within the Source 1
  entities competing for the same record; number of competing entities with a near-identical name;
  per-list blocking scores and how many lists retrieved the pair.
- **Other:** non-Latin-name flag, source flag (S2/S3).

**Model type:**
1. LightGBM binary classifier (255 leaves, early stopping on validation log-loss) trained on all
   candidates of 200,000 training entities (12.5M pairs).
2. Cross-encoder: `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M parameters)
   fine-tuned as a pair classifier on "name | address" of both records, using all true pairs plus
   the hardest blocking negatives. It re-scores pairs with LightGBM probability in the uncertain
   band, and the two are combined by a logistic blend of their logits fitted on validation.
3. Sibling-consistency stage (`src/stage2_cli.py`): for each candidate, best name / address
   similarity to the entity's other confident matches; a small LightGBM re-scores
   [logit(blended score), sibling features].

**Assignment:** each Source 2 / Source 3 record is kept only for its highest-scoring Source 1
entity, then the threshold is applied.

**Threshold selection method:** macro F_0.5 maximization on a held-out 20% validation split of the
training entities (seeded), with the one-entity-per-record assignment applied during the search.

---

## 5. Results & Error Analysis

- **F_0.5 score (macro, validation, US + India):**

  | stage | validation F_0.5 |
  |---|---|
  | LightGBM (v5) | 0.9627 |
  | + cross-encoder blend | 0.9739 |
  | + sibling-consistency stage | **0.9760** |

  Upper bound with a perfect classifier on our candidates (blocking ceiling): about 0.983.
- **Common false positives (wrong merges):** records that copy an entity's street with a different
  house or flat number ("A/1002" vs "A/100"), records with one distinctive name word swapped
  ("Amical (France) Ecole" vs "Amical (France) Federation"), and same-name businesses at a
  different branch when the other record has no address.
- **Common false negatives (missed matches):** Indic-script names with a heavily shortened address
  (only house number, city and state), invented brand names over a partially rewritten address,
  and records never retrieved by blocking (about 3.7% of true pairs).

---

## 6. Conclusion

A carefully measured blocking stage, scalable string and context features, and a cross-encoder used
only where it adds information took validation F_0.5 from 0.834 (first baseline) to 0.976 on a 15GB
laptop. The largest remaining gains lie in blocking recall and in French data, which has no labels
in training.

---

## Appendix

### A. Code Artefacts

Source lives in `code/business_entity_resolution/src/`:

| module | role |
|---|---|
| `normalize_v3.py` | text normalization, legal-form split, region canonicalization, phonetic skeletons |
| `blocking_v3.py` | hashed-token CSR matrices, per-country top-k cosine search, multi-list union |
| `features_v3.py` | record tables, vectorized pair features, global context features |
| `pipeline_v3.py` | `block`, `train`, `infer`, `finalize` subcommands |
| `cross_encoder.py` | cross-encoder data preparation, fine-tuning, scoring |
| `blend_cli.py` | LightGBM / cross-encoder blend fit and apply |
| `stage2_cli.py` | sibling-consistency stage fit and apply |
| `infer.py`, `metrics.py`, `utils_io.py` | assignment rule, F_0.5 metric, TSV I/O |

Reproduction (from `code/business_entity_resolution/`, dataset under `student_resource/dataset/`):

```bash
python -m src.pipeline_v3 block --multi --split train --dataset-dir DATA/train --out cands_train.npz
python -m src.pipeline_v3 block --multi --split test  --dataset-dir DATA/test  --out cands_test.npz
python -m src.pipeline_v3 train --dataset-dir DATA/train --candidates cands_train.npz --model-path model.joblib
python -m src.pipeline_v3 infer --dataset-dir DATA/test --candidates cands_test.npz --model-path model.joblib --output-dir out_lgb
python -m src.cross_encoder prepare --dataset-dir DATA/train --candidates cands_train.npz --out-dir ce
python -m src.cross_encoder train --data-dir ce --model-dir ce/model
python -m src.blend_cli fit --dataset-dir DATA/train --val-scores model.joblib.val.npz --model-dir ce/model --params-out blend.json
python -m src.blend_cli apply --dataset-dir DATA/test --scores-dir out_lgb --model-dir ce/model --params blend.json --output-dir out_ce
python -m src.stage2_cli fit --dataset-dir DATA/train --val-scores model.joblib.val.npz --model-dir ce/model --blend-params blend.json --model-out stage2.joblib
python -m src.stage2_cli apply --dataset-dir DATA/test --scores-dir out_lgb --ce-dir out_ce --stage2-model stage2.joblib --output-dir output
```

`output/` then holds `matching_results.tsv` and `candidate_pairs.tsv`.

### B. Additional Results

| version | change | validation F_0.5 | public leaderboard |
|---|---|---|---|
| v2 | pandas token/address blocking + embeddings, LightGBM | 0.834 | 0.789 |
| v2 + assignment rule | one Source 1 entity per record | – | 0.802 |
| v3 | sparse blocking, vectorized features | 0.9448 | 0.930 |
| v3b | + name/address context features | 0.9504 | 0.934 |
| v3b + cross-encoder | uncertain-pair re-scoring | 0.969 (realistic) | 0.943 |
| v5 + cross-encoder + stage 2 | multi-list blocking, number/skeleton features, region canonicalization, sibling stage | 0.976 (realistic) | **0.953** |

"Realistic" validation removes false positives whose true owner lies outside the validation sample;
on the test set every entity competes for its records, so those errors do not occur there.
