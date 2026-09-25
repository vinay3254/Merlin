# Kinfolk — Business Entity Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full entity-resolution pipeline (codename **Kinfolk**) that
matches noisy business records across 3 sources, producing
`output/matching_results.tsv` and `output/candidate_pairs.tsv` per the
Amazon ML Challenge 2026 format, scored on macro-averaged F_0.5.

**Architecture:** Normalize text → multi-strategy blocking (token index +
address prefix + embedding kNN) → per-pair similarity feature engineering →
LightGBM binary classifier → threshold-tuned per-entity assignment → TSV
output. Every stage is a small, independently testable module under
`code/business_entity_resolution/src/`.

**Tech Stack:** Python, pandas, numpy, rapidfuzz (MIT-licensed edit
distance — not python-Levenshtein, which is GPL), sentence-transformers
(`all-MiniLM-L6-v2`, ~22M params, for blocking and features only),
faiss-cpu (approximate nearest neighbor for embedding blocking at real
scale), lightgbm (MIT), joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-business-entity-resolution-design.md`

## Global Constraints

- All output TSVs are tab-separated; ID lists are comma-separated, no
  quoting, no duplicates within a list.
- `country` is an open string set — never hardcode, filter, or one-hot to a
  fixed list of country values anywhere in the pipeline.
- No external data/API lookups (geocoding, business registries, internet
  augmentation) anywhere in the pipeline.
- Final matching-stage model must be MIT/Apache-2.0 licensed and ≤8B
  parameters (LightGBM satisfies this trivially).
- Every Source-1 test entity must get exactly one row in
  `matching_results.tsv`, empty `matched_entity_ids` allowed (singletons).
- `matched_entity_ids` / `candidate_entity_ids` must only reference S2-/S3-
  IDs present in the test set — never S1 IDs, never IDs absent from the
  test source files.
- `matching_results.tsv` matches must be a subset of `candidate_pairs.tsv`
  candidates for the same `source1_entity_id`.
- Real dataset is in `student_resource/dataset/{train,test}/*.tsv`: 2.2M-5.3M
  rows per source file, ~2.5GB total. This rules out O(n×m) operations across
  full source pairs and `DataFrame.iterrows()` loops over a full source
  file — blocking uses vectorized joins (token/address) or a FAISS ANN index
  (embeddings), never a dense similarity matrix or a Python loop over
  millions of rows. Code that touches dataset paths must fail fast with a
  clear error message if a file is missing — never silently substitute mock
  data.
- Unit tests in this plan use small in-memory fixture DataFrames (1-5 rows),
  which pass regardless of whether an implementation is vectorized — they
  verify behavioral correctness, not scale. The scale requirement above is
  enforced by writing the specified vectorized/joined/FAISS code, not by a
  test assertion. Follow each task's Step 3 code as given; it is written the
  way it is specifically to be correct at real data volumes.

## Review Focus

- S1 entity with genuinely zero true matches (singleton): must appear in
  output with an empty `matched_entity_ids` cell, not be dropped or given a
  spurious match — covered in Task 10.
- Duplicate candidate/match IDs collapsing into one list: union of multiple
  blocking strategies or repeated kNN hits must dedupe before ever reaching
  a TSV row — covered in Tasks 5 and 1.
- Unseen `country` value at inference time (e.g. "France", never seen in
  training): normalization and country-match feature must handle it as an
  ordinary string, not crash or silently zero it out — covered in Tasks 2
  and 7.
- Candidate/match ID accidentally referencing a Source-1 record or an ID
  absent from the test files: blocking must only ever draw from S2/S3
  pools — covered in Task 4.
- Address record missing components (no PIN/state, landmark-only): numeric-
  token-overlap and prefix-key logic must degrade to "no signal" (0.0 /
  empty match), not raise on empty token sets — covered in Tasks 4 and 6.

---

## Task 1: Project scaffolding + TSV I/O utilities

**Files:**
- Create: `code/business_entity_resolution/src/__init__.py`
- Create: `code/business_entity_resolution/src/utils_io.py`
- Create: `code/business_entity_resolution/tests/__init__.py`
- Create: `code/business_entity_resolution/tests/test_utils_io.py`
- Create: `code/business_entity_resolution/requirements.txt`
- Create: `code/business_entity_resolution/pytest.ini`

**Interfaces:**
- Produces:
  - `read_source_tsv(path: str) -> pandas.DataFrame` — columns
    `entity_id, business_name, business_address, country`. Raises
    `FileNotFoundError` with a message naming the missing path if `path`
    doesn't exist.
  - `read_ground_truth(path: str) -> pandas.DataFrame` — columns
    `source1_entity_id, matched_entity_ids` (string, may be empty).
  - `parse_id_list(cell: str) -> list[str]` — splits a comma-separated cell,
    returns `[]` for empty/NaN.
  - `join_id_list(ids: list[str]) -> str` — sorts, dedupes (via `set()`),
    joins with `,`. Silent dedupe, not an exception: legitimate call sites
    (blocking-strategy union, per-entity match assignment) can hand it a
    list with incidental repeats, and the binding output constraint is "no
    duplicate IDs in the output list" — dedupe satisfies that regardless of
    input, where raising would make the pipeline brittle against a benign
    overlap.
  - `write_id_mapping_tsv(rows: dict[str, list[str]], path: str, id_col: str, list_col: str) -> None` —
    writes one row per key of `rows`, tab-separated, using `join_id_list`
    for the list column. Used for both `matching_results.tsv` (id_col=
    `source1_entity_id`, list_col=`matched_entity_ids`) and
    `candidate_pairs.tsv` (list_col=`candidate_entity_ids`).

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_utils_io.py
import pandas as pd
import pytest

from src.utils_io import (
    read_source_tsv,
    read_ground_truth,
    parse_id_list,
    join_id_list,
    write_id_mapping_tsv,
)


def test_read_source_tsv_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.tsv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        read_source_tsv(str(missing))


def test_read_source_tsv_parses_columns(tmp_path):
    p = tmp_path / "source1.tsv"
    p.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Corp\t123 Main St\tUS\n"
    )
    df = read_source_tsv(str(p))
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert df.iloc[0]["entity_id"] == "S1-00001"


def test_read_ground_truth_handles_empty_matches(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047,S3-00812\n"
        "S1-00002\t\n"
    )
    df = read_ground_truth(str(p))
    assert df.iloc[1]["matched_entity_ids"] in ("", None) or pd.isna(df.iloc[1]["matched_entity_ids"])


def test_parse_id_list_splits_and_handles_empty():
    assert parse_id_list("S2-00047,S3-00812") == ["S2-00047", "S3-00812"]
    assert parse_id_list("") == []
    assert parse_id_list(None) == []


def test_join_id_list_sorts_and_dedupes():
    assert join_id_list(["S3-00812", "S2-00047", "S2-00047"]) == "S2-00047,S3-00812"


def test_join_id_list_rejects_true_duplicates_from_caller():
    # caller passed the same id via two different code paths -- must not
    # silently collapse without the caller knowing; join_id_list itself
    # dedupes, but callers that need to detect a bug should check length
    # before calling. This test documents the dedupe behavior explicitly.
    result = join_id_list(["S2-00047", "S2-00047", "S3-00001"])
    assert result == "S2-00047,S3-00001"


def test_write_id_mapping_tsv_writes_expected_format(tmp_path):
    out = tmp_path / "matching_results.tsv"
    rows = {
        "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
        "S1-00002": ["S3-00004"],
        "S1-00003": [],
    }
    write_id_mapping_tsv(rows, str(out), id_col="source1_entity_id", list_col="matched_entity_ids")
    text = out.read_text()
    lines = text.strip("\n").split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert "S1-00001\tS2-00047,S2-00193,S3-00812" in lines
    assert "S1-00003\t" in text  # empty list still gets a row
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_utils_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils_io'`

- [ ] **Step 3: Write `requirements.txt` and `pytest.ini`**

```
# code/business_entity_resolution/requirements.txt
pandas==2.2.3
numpy==1.26.4
rapidfuzz==3.10.1
sentence-transformers==3.3.1
faiss-cpu==1.9.0
lightgbm==4.5.0
joblib==1.4.2
pytest==8.3.3
```

```ini
# code/business_entity_resolution/pytest.ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 4: Write minimal implementation**

```python
# code/business_entity_resolution/src/utils_io.py
import os
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_source_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected source TSV at {path} but it does not exist. "
            "Place the challenge dataset under dataset/train or dataset/test "
            "before running this pipeline."
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df.fillna("")


def read_ground_truth(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected ground truth TSV at {path} but it does not exist."
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in GROUND_TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def parse_id_list(cell) -> list:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == "":
        return []
    return [x for x in str(cell).split(",") if x]


def join_id_list(ids: list) -> str:
    return ",".join(sorted(set(ids)))


def write_id_mapping_tsv(rows: dict, path: str, id_col: str, list_col: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for key in sorted(rows.keys()):
            f.write(f"{key}\t{join_id_list(rows[key])}\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_utils_io.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add code/business_entity_resolution/src/utils_io.py \
        code/business_entity_resolution/src/__init__.py \
        code/business_entity_resolution/tests/test_utils_io.py \
        code/business_entity_resolution/tests/__init__.py \
        code/business_entity_resolution/requirements.txt \
        code/business_entity_resolution/pytest.ini
git commit -m "feat: add TSV I/O utilities with fail-fast file checks"
```

---

## Task 2: Text normalization

**Files:**
- Create: `code/business_entity_resolution/src/normalize.py`
- Create: `code/business_entity_resolution/tests/test_normalize.py`

**Interfaces:**
- Consumes: nothing (pure string functions)
- Produces:
  - `ABBREVIATION_MAP: dict[str, str]` — module-level constant mapping
    lowercase abbreviated tokens to expanded forms (e.g. `"corp": "corporation"`,
    `"pvt": "private"`, `"ltd": "limited"`, `"rd": "road"`, `"st": "street"`,
    `"&": "and"`).
  - `normalize_text(text: str) -> str` — lowercase, strip punctuation except
    intra-word hyphens/digits, expand abbreviations token-by-token via
    `ABBREVIATION_MAP`, collapse whitespace. Returns `""` for `None`/empty
    input, never raises.
  - `tokenize(normalized_text: str) -> list[str]` — whitespace split of
    already-normalized text, drops empty tokens.

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_normalize.py
from src.normalize import normalize_text, tokenize, ABBREVIATION_MAP


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("Acme, Corp.!") == "acme corporation"


def test_normalize_expands_common_abbreviations():
    assert normalize_text("Sharma & Sons Pvt Ltd") == "sharma and sons private limited"
    assert normalize_text("123 MG Rd") == "123 mg road"
    assert normalize_text("456 Park St") == "456 park street"


def test_normalize_handles_none_and_empty_without_raising():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


def test_normalize_collapses_whitespace():
    assert normalize_text("Acme   Corp") == "acme corporation"


def test_normalize_handles_unseen_country_style_free_text():
    # "France"-style free text with accents/punctuation must not crash
    assert normalize_text("Café de Paris") == "cafe de paris" or normalize_text("Café de Paris") != ""


def test_tokenize_splits_normalized_text():
    assert tokenize(normalize_text("Acme Corp")) == ["acme", "corporation"]


def test_tokenize_handles_empty_string():
    assert tokenize("") == []


def test_abbreviation_map_is_lowercase_keys():
    assert all(k == k.lower() for k in ABBREVIATION_MAP)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/normalize.py
import re
import unicodedata

ABBREVIATION_MAP = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "&": "and",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "apt": "apartment",
    "bldg": "building",
    "no": "number",
}

_PUNCT_RE = re.compile(r"[^\w\s&-]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    expanded = [ABBREVIATION_MAP.get(tok, tok) for tok in tokens]
    return _WS_RE.sub(" ", " ".join(expanded)).strip()


def tokenize(normalized_text: str) -> list:
    if not normalized_text:
        return []
    return [t for t in normalized_text.split(" ") if t]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_normalize.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/normalize.py \
        code/business_entity_resolution/tests/test_normalize.py
git commit -m "feat: add text normalization with abbreviation expansion"
```

---

## Task 3: F_0.5 macro-average metric

**Files:**
- Create: `code/business_entity_resolution/src/metrics.py`
- Create: `code/business_entity_resolution/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `f_beta_score(predicted: set[str], actual: set[str], beta: float = 0.5) -> float` —
    per-entity score. Empty `actual` and empty `predicted` → `1.0`. Empty
    `actual` and non-empty `predicted` → `0.0`. Non-empty `actual` and empty
    `predicted` → `0.0` (recall 0). Otherwise standard F_beta on
    precision/recall over set overlap.
  - `macro_f_beta(predictions: dict[str, set[str]], actuals: dict[str, set[str]], beta: float = 0.5) -> float` —
    averages `f_beta_score` over every key in `actuals`, using `predictions.get(key, set())`
    when a key is absent from `predictions`.

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_metrics.py
import math
from src.metrics import f_beta_score, macro_f_beta


def test_f_beta_score_matches_spec_example():
    predicted = {"S2-00047", "S2-00193", "S3-00812"}
    actual = {"S2-00047", "S3-00812"}
    score = f_beta_score(predicted, actual, beta=0.5)
    assert math.isclose(score, 0.714, abs_tol=0.001)


def test_f_beta_score_singleton_correct_empty_prediction_is_one():
    assert f_beta_score(set(), set(), beta=0.5) == 1.0


def test_f_beta_score_singleton_false_merge_is_zero():
    assert f_beta_score({"S2-00001"}, set(), beta=0.5) == 0.0


def test_f_beta_score_missed_all_matches_is_zero():
    assert f_beta_score(set(), {"S2-00001"}, beta=0.5) == 0.0


def test_f_beta_score_perfect_match_is_one():
    assert f_beta_score({"S2-00001", "S3-00002"}, {"S2-00001", "S3-00002"}, beta=0.5) == 1.0


def test_macro_f_beta_averages_across_entities_including_singletons():
    predictions = {
        "S1-00001": {"S2-00047", "S3-00812"},
        "S1-00002": set(),
    }
    actuals = {
        "S1-00001": {"S2-00047", "S3-00812"},
        "S1-00002": set(),
    }
    assert macro_f_beta(predictions, actuals) == 1.0


def test_macro_f_beta_uses_empty_set_for_missing_prediction_key():
    predictions = {}
    actuals = {"S1-00001": {"S2-00047"}}
    assert macro_f_beta(predictions, actuals) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/metrics.py
def f_beta_score(predicted: set, actual: set, beta: float = 0.5) -> float:
    if not actual and not predicted:
        return 1.0
    if not predicted or not actual:
        return 0.0
    true_positives = len(predicted & actual)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(predicted)
    recall = true_positives / len(actual)
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(predictions: dict, actuals: dict, beta: float = 0.5) -> float:
    if not actuals:
        return 0.0
    scores = [
        f_beta_score(predictions.get(key, set()), actual, beta=beta)
        for key, actual in actuals.items()
    ]
    return sum(scores) / len(scores)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/metrics.py \
        code/business_entity_resolution/tests/test_metrics.py
git commit -m "feat: add F_0.5 macro-average metric matching challenge spec"
```

---

## Task 4: Token-overlap and address-prefix blocking

**Files:**
- Create: `code/business_entity_resolution/src/blocking.py`
- Create: `code/business_entity_resolution/tests/test_blocking.py`

**Interfaces:**
- Consumes: `normalize_text`, `tokenize` from `src.normalize`
- Produces:
  - `STOPWORDS: set[str]` — tokens excluded from index keys (e.g. `"the"`,
    "and", "inc", "corporation", "limited", "company", "of").
  - `MAX_TOKEN_DOC_FREQ: int` — default `5000`. A token appearing in more
    than this many `other_df` records is dropped before joining (a
    block-size cap — without it, a common token shared by hundreds of
    thousands of records would blow the join up into a useless giant
    candidate set at real data scale).
  - `build_token_index(df, id_col="entity_id", text_col="business_name") -> dict[str, set[str]]` —
    token → set of entity_ids, skipping stopword tokens. Kept as a small
    convenience wrapper for callers that want a plain dict; internally
    implemented via the same exploded-table approach as
    `token_overlap_candidates` (below), not a per-row Python loop.
  - `token_overlap_candidates(s1_df, other_df, max_doc_freq=MAX_TOKEN_DOC_FREQ) -> dict[str, set[str]]` —
    for each S1 `entity_id`, the union of `other_df` entity_ids sharing at
    least one non-stopword, non-capped normalized-name token. Implemented as
    a pandas merge (join) between two long-format `(entity_id, token)`
    tables, never a Python loop over `other_df`'s rows. `other_df` entity_ids
    are always taken verbatim from `other_df["entity_id"]`, so only S2/S3 ids
    ever appear (never S1 ids, since S1 is never passed as `other_df`). Every
    `entity_id` in `s1_df` is a key in the result (empty set if no match).
  - `address_prefix_candidates(s1_df, other_df, prefix_len=3) -> dict[str, set[str]]` —
    keys on the first `prefix_len` normalized address tokens (joined into one
    string key); entities with fewer than `prefix_len` tokens (missing
    components) key on whatever tokens they have, and entities with zero
    address tokens produce no key and get no candidates from this strategy
    (never raise). Implemented as a pandas merge on the prefix-key column,
    never a Python loop over `other_df`'s rows.

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_blocking.py
import pandas as pd
from src.blocking import build_token_index, token_overlap_candidates, address_prefix_candidates


def _df(rows):
    return pd.DataFrame(rows)


def test_token_overlap_candidates_matches_shared_significant_token():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Sharma Traders", "business_address": "", "country": "India"}])
    s2 = _df([
        {"entity_id": "S2-00001", "business_name": "Sharma Traders Pvt Ltd", "business_address": "", "country": "India"},
        {"entity_id": "S2-00002", "business_name": "Totally Unrelated Bakery", "business_address": "", "country": "India"},
    ])
    result = token_overlap_candidates(s1, s2)
    assert result["S1-00001"] == {"S2-00001"}


def test_token_overlap_candidates_never_includes_source1_ids():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    result = token_overlap_candidates(s1, s2)
    assert all(cid.startswith("S2-") or cid.startswith("S3-") for ids in result.values() for cid in ids)


def test_token_overlap_candidates_no_match_gives_empty_set():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Zephyr Logistics", "business_address": "", "country": "US"}])
    result = token_overlap_candidates(s1, s2)
    assert result["S1-00001"] == set()


def test_address_prefix_candidates_handles_missing_address_without_raising():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    result = address_prefix_candidates(s1, s2)
    assert result["S1-00001"] == set()


def test_address_prefix_candidates_matches_shared_prefix():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "", "business_address": "123 MG Road Bangalore", "country": "India"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "", "business_address": "123 MG Road Near SBI ATM", "country": "India"}])
    result = address_prefix_candidates(s1, s2, prefix_len=3)
    assert result["S1-00001"] == {"S2-00001"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_blocking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.blocking'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/blocking.py
import pandas as pd

from src.normalize import normalize_text, tokenize

STOPWORDS = {
    "the", "and", "of", "inc", "incorporated", "corporation", "corp",
    "limited", "ltd", "company", "co", "private", "pvt", "llc", "llp",
}

MAX_TOKEN_DOC_FREQ = 5000


def _name_tokens_table(df, id_col_name: str, text_col: str = "business_name") -> pd.DataFrame:
    # .map (one Python call per row) instead of iterrows (no per-row Series
    # construction); explode() is a vectorized pandas op, not a Python loop.
    token_lists = df[text_col].map(
        lambda t: [tok for tok in tokenize(normalize_text(t)) if tok not in STOPWORDS]
    )
    table = pd.DataFrame({id_col_name: df["entity_id"].values, "token": token_lists})
    return table.explode("token").dropna(subset=["token"])


def build_token_index(df, id_col="entity_id", text_col="business_name") -> dict:
    table = _name_tokens_table(df, id_col_name="entity_id", text_col=text_col)
    index = {}
    for token, group in table.groupby("token")["entity_id"]:
        index[token] = set(group)
    return index


def token_overlap_candidates(s1_df, other_df, max_doc_freq: int = MAX_TOKEN_DOC_FREQ) -> dict:
    s1_tokens = _name_tokens_table(s1_df, id_col_name="entity_id_s1")
    other_tokens = _name_tokens_table(other_df, id_col_name="entity_id_other")

    doc_freq = other_tokens["token"].value_counts()
    allowed_tokens = set(doc_freq[doc_freq <= max_doc_freq].index)
    other_tokens = other_tokens[other_tokens["token"].isin(allowed_tokens)]

    result = {eid: set() for eid in s1_df["entity_id"]}
    if s1_tokens.empty or other_tokens.empty:
        return result

    merged = s1_tokens.merge(other_tokens, on="token")
    if merged.empty:
        return result

    grouped = merged.groupby("entity_id_s1")["entity_id_other"].agg(set)
    result.update(grouped.to_dict())
    return result


def _address_prefix_key_table(df, id_col_name: str, prefix_len: int) -> pd.DataFrame:
    keys = df["business_address"].map(
        lambda addr: " ".join(tokenize(normalize_text(addr))[:prefix_len]) or None
    )
    table = pd.DataFrame({id_col_name: df["entity_id"].values, "prefix": keys})
    return table.dropna(subset=["prefix"])


def address_prefix_candidates(s1_df, other_df, prefix_len: int = 3) -> dict:
    result = {eid: set() for eid in s1_df["entity_id"]}

    s1_keys = _address_prefix_key_table(s1_df, "entity_id_s1", prefix_len)
    other_keys = _address_prefix_key_table(other_df, "entity_id_other", prefix_len)
    if s1_keys.empty or other_keys.empty:
        return result

    merged = s1_keys.merge(other_keys, on="prefix")
    if merged.empty:
        return result

    grouped = merged.groupby("entity_id_s1")["entity_id_other"].agg(set)
    result.update(grouped.to_dict())
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_blocking.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/blocking.py \
        code/business_entity_resolution/tests/test_blocking.py
git commit -m "feat: add token-overlap and address-prefix blocking strategies"
```

---

## Task 5: Embedding kNN blocking + candidate union

**Files:**
- Modify: `code/business_entity_resolution/src/blocking.py` (append)
- Modify: `code/business_entity_resolution/tests/test_blocking.py` (append)

**Interfaces:**
- Consumes: `normalize_text` from `src.normalize`; `token_overlap_candidates`,
  `address_prefix_candidates` from this task's own module (Task 4)
- Produces:
  - `default_embedder(texts: list[str]) -> numpy.ndarray` — lazily imports
    `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` on first
    call, returns L2-normalized embeddings. Not unit tested directly (would
    require a network download); covered by injecting a fake embedder in
    tests of the function that consumes it.
  - `embedding_knn_candidates(s1_df, other_df, embedder=default_embedder, top_k=10, min_sim=0.5) -> dict[str, set[str]]` —
    `embedder` is any `callable(list[str]) -> numpy.ndarray` of unit vectors.
    **Partitions by the raw `country` string** (via `groupby`) before doing
    any similarity search: within each partition, embeds that partition's
    `business_name + " " + business_address` text and searches a FAISS
    index built from that partition's `other_df` rows only. This keeps both
    the embedding matrix and the FAISS index within available RAM/VRAM at
    real data scale (a country-partitioned index over ~1-5M rows instead of
    one index over 5M+ rows, and never a dense S1×other matrix). A partition
    with fewer than 1000 `other_df` rows uses an exact `faiss.IndexFlatIP`;
    larger partitions train a `faiss.IndexIVFFlat` (approximate, sub-linear
    search). Candidates are kept with similarity ≥ `min_sim`, up to `top_k`
    per S1 entity. S1 entities whose country has no matching `other_df`
    partition get an empty set (not an error) — this strategy simply
    contributes nothing for them; strategies 1-2 are not country-partitioned
    and remain reachable.
  - `union_candidates(*candidate_dicts: dict[str, set[str]]) -> dict[str, set[str]]` —
    merges any number of `entity_id -> set(candidate_ids)` dicts, union per
    key, dedupe guaranteed by set semantics.
  - `candidates_to_rows(union_dict: dict[str, set[str]]) -> dict[str, list[str]]` —
    converts each value to a sorted `list` (ready for
    `utils_io.write_id_mapping_tsv`).

- [ ] **Step 1: Write the failing tests**

```python
# appended to code/business_entity_resolution/tests/test_blocking.py
import numpy as np
from src.blocking import embedding_knn_candidates, union_candidates, candidates_to_rows


def _fake_embedder(texts):
    # deterministic stub: identical texts -> identical vectors, no network/model needed
    vocab = {}
    vectors = []
    for t in texts:
        key = t.strip().lower()
        if key not in vocab:
            vocab[key] = len(vocab)
        idx = vocab[key]
        vec = np.zeros(8)
        vec[idx % 8] = 1.0
        vectors.append(vec)
    return np.array(vectors)


def test_embedding_knn_candidates_matches_identical_text():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "acme traders", "business_address": "123 main st", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "acme traders", "business_address": "123 main st", "country": "US"}])
    result = embedding_knn_candidates(s1, s2, embedder=_fake_embedder, top_k=5, min_sim=0.99)
    assert result["S1-00001"] == {"S2-00001"}


def test_embedding_knn_candidates_respects_min_sim_threshold():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "alpha", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "zzz completely different", "business_address": "", "country": "US"}])
    result = embedding_knn_candidates(s1, s2, embedder=_fake_embedder, top_k=5, min_sim=0.99)
    assert result["S1-00001"] == set()


def test_union_candidates_merges_and_dedupes():
    a = {"S1-00001": {"S2-00001", "S2-00002"}}
    b = {"S1-00001": {"S2-00002", "S3-00001"}}
    merged = union_candidates(a, b)
    assert merged["S1-00001"] == {"S2-00001", "S2-00002", "S3-00001"}


def test_union_candidates_handles_key_present_in_only_one_dict():
    a = {"S1-00001": {"S2-00001"}}
    b = {"S1-00002": {"S3-00001"}}
    merged = union_candidates(a, b)
    assert merged["S1-00001"] == {"S2-00001"}
    assert merged["S1-00002"] == {"S3-00001"}


def test_candidates_to_rows_produces_sorted_lists():
    union = {"S1-00001": {"S2-00002", "S2-00001"}}
    rows = candidates_to_rows(union)
    assert rows["S1-00001"] == ["S2-00001", "S2-00002"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_blocking.py -v`
Expected: FAIL with `ImportError: cannot import name 'embedding_knn_candidates'`

- [ ] **Step 3: Write minimal implementation**

```python
# appended to code/business_entity_resolution/src/blocking.py
from collections import defaultdict

import numpy as np

FLAT_INDEX_THRESHOLD = 1000  # below this many rows, exact search is cheap enough


def default_embedder(texts: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = getattr(default_embedder, "_model", None)
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        default_embedder._model = model
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors)


def _combined_texts(df) -> list:
    names = df["business_name"].map(normalize_text)
    addrs = df["business_address"].map(normalize_text)
    return [f"{n} {a}".strip() for n, a in zip(names, addrs)]


def _search_partition(s1_df, other_df, embedder, top_k: int, min_sim: float) -> dict:
    import faiss

    s1_ids = s1_df["entity_id"].tolist()
    other_ids = other_df["entity_id"].tolist()

    s1_vecs = np.ascontiguousarray(embedder(_combined_texts(s1_df)), dtype="float32")
    other_vecs = np.ascontiguousarray(embedder(_combined_texts(other_df)), dtype="float32")
    dim = other_vecs.shape[1]

    if len(other_ids) < FLAT_INDEX_THRESHOLD:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = max(1, min(int(len(other_ids) ** 0.5), 4096))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(other_vecs)
        index.nprobe = min(nlist, 10)
    index.add(other_vecs)

    k = min(top_k, len(other_ids))
    sims, indices = index.search(s1_vecs, k)

    result = {}
    for i, s1_id in enumerate(s1_ids):
        candidates = {
            other_ids[idx]
            for idx, sim in zip(indices[i], sims[i])
            if idx != -1 and sim >= min_sim
        }
        result[s1_id] = candidates
    return result


def embedding_knn_candidates(s1_df, other_df, embedder=default_embedder, top_k: int = 10, min_sim: float = 0.5) -> dict:
    result = {eid: set() for eid in s1_df["entity_id"]}
    if len(s1_df) == 0 or len(other_df) == 0:
        return result

    s1_country = s1_df["country"].fillna("")
    other_country = other_df["country"].fillna("")
    for country, s1_group in s1_df.groupby(s1_country):
        other_group = other_df[other_country == country]
        if len(other_group) == 0:
            continue
        result.update(_search_partition(s1_group, other_group, embedder, top_k, min_sim))
    return result


def union_candidates(*candidate_dicts) -> dict:
    merged = defaultdict(set)
    for d in candidate_dicts:
        for key, ids in d.items():
            merged[key] |= ids
    return dict(merged)


def candidates_to_rows(union_dict: dict) -> dict:
    return {key: sorted(ids) for key, ids in union_dict.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_blocking.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/blocking.py \
        code/business_entity_resolution/tests/test_blocking.py
git commit -m "feat: add embedding kNN blocking and candidate union"
```

---

## Task 6: Name and address similarity features

**Files:**
- Create: `code/business_entity_resolution/src/features.py`
- Create: `code/business_entity_resolution/tests/test_features.py`

**Interfaces:**
- Consumes: `normalize_text`, `tokenize` from `src.normalize`
- Produces:
  - `levenshtein_ratio(a: str, b: str) -> float` — `rapidfuzz.fuzz.ratio(a, b) / 100.0`,
    `0.0` if either input is empty.
  - `token_jaccard(a_tokens: list[str], b_tokens: list[str]) -> float` —
    `0.0` if either list is empty.
  - `common_token_count(a_tokens: list[str], b_tokens: list[str]) -> int`
  - `sorted_tokens_exact_match(a_tokens: list[str], b_tokens: list[str]) -> int` —
    `1` if `sorted(a_tokens) == sorted(b_tokens)` and both non-empty, else `0`.
  - `numeric_token_overlap(a_text: str, b_text: str) -> float` — Jaccard over
    digit-only substrings extracted from each raw text; `0.0` if either side
    has no numeric tokens (never raises on empty input).
  - `name_address_string_features(name_a, addr_a, name_b, addr_b) -> dict` —
    bundles all of the above into a flat feature dict with keys:
    `name_levenshtein`, `name_jaccard`, `name_common_tokens`,
    `name_exact_normalized_match`, `addr_levenshtein`, `addr_jaccard`,
    `addr_sorted_token_match`, `addr_numeric_overlap`.

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_features.py
from src.features import (
    levenshtein_ratio,
    token_jaccard,
    common_token_count,
    sorted_tokens_exact_match,
    numeric_token_overlap,
    name_address_string_features,
)


def test_levenshtein_ratio_identical_strings_is_one():
    assert levenshtein_ratio("acme corp", "acme corp") == 1.0


def test_levenshtein_ratio_empty_string_is_zero():
    assert levenshtein_ratio("", "acme") == 0.0
    assert levenshtein_ratio("acme", "") == 0.0


def test_token_jaccard_partial_overlap():
    assert token_jaccard(["acme", "traders"], ["acme", "logistics"]) == 1 / 3


def test_token_jaccard_empty_list_is_zero():
    assert token_jaccard([], ["acme"]) == 0.0
    assert token_jaccard([], []) == 0.0


def test_common_token_count():
    assert common_token_count(["a", "b", "c"], ["b", "c", "d"]) == 2


def test_sorted_tokens_exact_match_handles_reordering():
    assert sorted_tokens_exact_match(["main", "st", "123"], ["123", "main", "st"]) == 1
    assert sorted_tokens_exact_match(["main", "st"], ["oak", "ave"]) == 0
    assert sorted_tokens_exact_match([], []) == 0


def test_numeric_token_overlap_handles_missing_numbers_without_raising():
    assert numeric_token_overlap("Near SBI ATM", "Landmark reference only") == 0.0
    assert numeric_token_overlap("", "") == 0.0


def test_numeric_token_overlap_matches_shared_numbers():
    assert numeric_token_overlap("123 Main St 560001", "123 Main Road") > 0.0


def test_name_address_string_features_returns_all_keys():
    feats = name_address_string_features("Acme Corp", "123 Main St", "Acme Corporation", "123 Main Street")
    expected_keys = {
        "name_levenshtein", "name_jaccard", "name_common_tokens",
        "name_exact_normalized_match", "addr_levenshtein", "addr_jaccard",
        "addr_sorted_token_match", "addr_numeric_overlap",
    }
    assert set(feats.keys()) == expected_keys
    assert feats["name_exact_normalized_match"] == 1  # both normalize to "acme corporation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/features.py
import re

from rapidfuzz import fuzz

from src.normalize import normalize_text, tokenize

_DIGIT_RE = re.compile(r"\d+")


def levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def token_jaccard(a_tokens: list, b_tokens: list) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set, b_set = set(a_tokens), set(b_tokens)
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def common_token_count(a_tokens: list, b_tokens: list) -> int:
    return len(set(a_tokens) & set(b_tokens))


def sorted_tokens_exact_match(a_tokens: list, b_tokens: list) -> int:
    if not a_tokens or not b_tokens:
        return 0
    return int(sorted(a_tokens) == sorted(b_tokens))


def numeric_token_overlap(a_text: str, b_text: str) -> float:
    a_nums = set(_DIGIT_RE.findall(a_text or ""))
    b_nums = set(_DIGIT_RE.findall(b_text or ""))
    if not a_nums or not b_nums:
        return 0.0
    union = a_nums | b_nums
    return len(a_nums & b_nums) / len(union)


def name_address_string_features(name_a, addr_a, name_b, addr_b) -> dict:
    norm_name_a, norm_name_b = normalize_text(name_a), normalize_text(name_b)
    norm_addr_a, norm_addr_b = normalize_text(addr_a), normalize_text(addr_b)
    name_tokens_a, name_tokens_b = tokenize(norm_name_a), tokenize(norm_name_b)
    addr_tokens_a, addr_tokens_b = tokenize(norm_addr_a), tokenize(norm_addr_b)

    return {
        "name_levenshtein": levenshtein_ratio(norm_name_a, norm_name_b),
        "name_jaccard": token_jaccard(name_tokens_a, name_tokens_b),
        "name_common_tokens": common_token_count(name_tokens_a, name_tokens_b),
        "name_exact_normalized_match": int(bool(norm_name_a) and norm_name_a == norm_name_b),
        "addr_levenshtein": levenshtein_ratio(norm_addr_a, norm_addr_b),
        "addr_jaccard": token_jaccard(addr_tokens_a, addr_tokens_b),
        "addr_sorted_token_match": sorted_tokens_exact_match(addr_tokens_a, addr_tokens_b),
        "addr_numeric_overlap": numeric_token_overlap(addr_a, addr_b),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_features.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/features.py \
        code/business_entity_resolution/tests/test_features.py
git commit -m "feat: add name/address string similarity features"
```

---

## Task 7: Embedding, country, structural features + feature matrix builder

**Files:**
- Modify: `code/business_entity_resolution/src/features.py` (append)
- Modify: `code/business_entity_resolution/tests/test_features.py` (append)

**Interfaces:**
- Consumes: `name_address_string_features` from this module (Task 6);
  `normalize_text`, `tokenize` from `src.normalize`
- Produces:
  - `country_match(country_a: str, country_b: str) -> int` — case-insensitive
    stripped exact-match flag; `0` if either is empty; treats any string
    value equally (no fixed country set).
  - `structural_features(name_a, name_b, addr_a, addr_b) -> dict` — keys
    `name_length_diff`, `name_token_count_diff`, `addr_length_diff`,
    `addr_token_count_diff` (all non-negative ints).
  - `embedding_cosine_features(vec_a: numpy.ndarray, vec_b: numpy.ndarray) -> dict` —
    key `embedding_cosine` = dot product (vectors assumed unit-normalized,
    consistent with `blocking.default_embedder`'s output); `0.0` if either
    vector is `None`.
  - `build_pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b, embed_a=None, embed_b=None) -> dict` —
    merges `name_address_string_features`, `country_match` (under key
    `country_match`), `structural_features`, and `embedding_cosine_features`
    into one flat dict — this is the row-level function
    Task 9's `build_feature_matrix` will call per pair.

- [ ] **Step 1: Write the failing tests**

```python
# appended to code/business_entity_resolution/tests/test_features.py
import numpy as np
from src.features import (
    country_match,
    structural_features,
    embedding_cosine_features,
    build_pair_features,
)


def test_country_match_exact_and_case_insensitive():
    assert country_match("US", "us") == 1
    assert country_match("India", "US") == 0


def test_country_match_handles_unseen_country_value_like_france():
    assert country_match("France", "France") == 1
    assert country_match("France", "US") == 0


def test_country_match_empty_is_zero():
    assert country_match("", "US") == 0
    assert country_match("", "") == 0


def test_structural_features_computes_diffs():
    feats = structural_features("Acme Corp", "Acme", "123 Main St", "123 Main")
    assert feats["name_length_diff"] == len("Acme Corp") - len("Acme")
    assert feats["name_token_count_diff"] >= 0


def test_embedding_cosine_features_handles_none_vectors():
    assert embedding_cosine_features(None, None)["embedding_cosine"] == 0.0


def test_embedding_cosine_features_computes_dot_product():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert embedding_cosine_features(a, b)["embedding_cosine"] == 1.0


def test_build_pair_features_merges_all_feature_groups():
    feats = build_pair_features(
        "Acme Corp", "123 Main St", "US",
        "Acme Corporation", "123 Main Street", "US",
        embed_a=np.array([1.0, 0.0]), embed_b=np.array([1.0, 0.0]),
    )
    assert "name_levenshtein" in feats
    assert "country_match" in feats and feats["country_match"] == 1
    assert "name_length_diff" in feats
    assert "embedding_cosine" in feats and feats["embedding_cosine"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'country_match'`

- [ ] **Step 3: Write minimal implementation**

```python
# appended to code/business_entity_resolution/src/features.py
def country_match(country_a: str, country_b: str) -> int:
    a = (country_a or "").strip().lower()
    b = (country_b or "").strip().lower()
    if not a or not b:
        return 0
    return int(a == b)


def structural_features(name_a, name_b, addr_a, addr_b) -> dict:
    norm_name_a, norm_name_b = normalize_text(name_a), normalize_text(name_b)
    norm_addr_a, norm_addr_b = normalize_text(addr_a), normalize_text(addr_b)
    return {
        "name_length_diff": abs(len(norm_name_a) - len(norm_name_b)),
        "name_token_count_diff": abs(len(tokenize(norm_name_a)) - len(tokenize(norm_name_b))),
        "addr_length_diff": abs(len(norm_addr_a) - len(norm_addr_b)),
        "addr_token_count_diff": abs(len(tokenize(norm_addr_a)) - len(tokenize(norm_addr_b))),
    }


def embedding_cosine_features(vec_a, vec_b) -> dict:
    if vec_a is None or vec_b is None:
        return {"embedding_cosine": 0.0}
    return {"embedding_cosine": float(vec_a @ vec_b)}


def build_pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b, embed_a=None, embed_b=None) -> dict:
    feats = {}
    feats.update(name_address_string_features(name_a, addr_a, name_b, addr_b))
    feats["country_match"] = country_match(country_a, country_b)
    feats.update(structural_features(name_a, name_b, addr_a, addr_b))
    feats.update(embedding_cosine_features(embed_a, embed_b))
    return feats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_features.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/features.py \
        code/business_entity_resolution/tests/test_features.py
git commit -m "feat: add embedding/country/structural features and pair-feature builder"
```

---

## Task 8: Training pair construction and feature matrix

**Files:**
- Create: `code/business_entity_resolution/src/train.py`
- Create: `code/business_entity_resolution/tests/test_train.py`

**Interfaces:**
- Consumes: `union_candidates`, `token_overlap_candidates`,
  `address_prefix_candidates`, `embedding_knn_candidates`, `default_embedder`
  from `src.blocking`; `build_pair_features` from `src.features`;
  `parse_id_list` from `src.utils_io`
- Produces:
  - `build_training_pairs(s1_df, others_df, ground_truth_df, blocking_candidates, max_negatives_per_positive=10, seed=42) -> pandas.DataFrame` —
    `others_df` is the concatenation of source2/source3 training DataFrames.
    `blocking_candidates` is `dict[str, set[str]]` (S1 id → candidate ids)
    from the union of blocking strategies. Returns a DataFrame with columns
    `source1_entity_id, other_entity_id, label` where `label=1` for every
    id in that S1 entity's ground-truth matched set (added even if blocking
    missed it, so positives are never starved) and `label=0` for a sample of
    that entity's `blocking_candidates` minus its positives (hard negatives).
    **Negative sampling cap:** at real data scale (2.2M Source-1 entities),
    including every blocking-candidate false positive as a labeled row can
    produce tens of millions of negative rows. Cap negatives per entity to
    `max_negatives_per_positive * max(1, len(positives))` — deterministically
    sampled via `random.Random(seed)` when the candidate pool exceeds the
    cap — rather than including every candidate.
    Ground-truth parsing uses `zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"])`
    to build the id→matches dict, not `.iterrows()` (2.2M-row ground truth at
    real scale — `.iterrows()` reconstructs a pandas Series per row and is
    measurably slower for no benefit here).
  - `build_feature_matrix(pairs_df, s1_df, others_df, embed_lookup=None) -> pandas.DataFrame` —
    **merges** `pairs_df` against `s1_df`/`others_df` on entity_id (a single
    vectorized pandas merge each side, not a `.loc` lookup per row), then
    iterates the merged frame with `itertuples()` (not `iterrows()`) calling
    `build_pair_features` per row, and returns `pairs_df` columns plus one
    column per feature key. `embed_lookup` is an optional
    `dict[entity_id, numpy.ndarray]`; when `None`, embedding features are
    computed with `0.0` (embedding_cosine defaults via `embed_a=None`).
  - `train_model(feature_matrix_df, feature_columns) -> lightgbm.LGBMClassifier` —
    fits on `feature_matrix_df[feature_columns]` against `feature_matrix_df["label"]`.

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_train.py
import pandas as pd
from src.train import build_training_pairs, build_feature_matrix, train_model


def _df(rows):
    return pd.DataFrame(rows)


def test_build_training_pairs_includes_ground_truth_positives_even_if_blocking_missed_them():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    others = _df([
        {"entity_id": "S2-00001", "business_name": "Acme Inc", "business_address": "", "country": "US"},
        {"entity_id": "S2-00002", "business_name": "Zephyr", "business_address": "", "country": "US"},
    ])
    gt = _df([{"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001"}])
    blocking_candidates = {"S1-00001": {"S2-00002"}}  # blocking missed the true match, found a false one

    pairs = build_training_pairs(s1, others, gt, blocking_candidates)

    positive = pairs[(pairs["other_entity_id"] == "S2-00001")]
    negative = pairs[(pairs["other_entity_id"] == "S2-00002")]
    assert positive.iloc[0]["label"] == 1
    assert negative.iloc[0]["label"] == 0


def test_build_training_pairs_singleton_produces_no_positive_rows():
    s1 = _df([{"entity_id": "S1-00002", "business_name": "Solo", "business_address": "", "country": "US"}])
    others = _df([{"entity_id": "S2-00003", "business_name": "Unrelated", "business_address": "", "country": "US"}])
    gt = _df([{"source1_entity_id": "S1-00002", "matched_entity_ids": ""}])
    blocking_candidates = {"S1-00002": {"S2-00003"}}

    pairs = build_training_pairs(s1, others, gt, blocking_candidates)
    assert (pairs["label"] == 1).sum() == 0
    assert (pairs["label"] == 0).sum() == 1


def test_build_feature_matrix_produces_one_row_per_pair_with_feature_columns():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "123 Main St", "country": "US"}])
    others = _df([{"entity_id": "S2-00001", "business_name": "Acme Inc", "business_address": "123 Main St", "country": "US"}])
    pairs = _df([{"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "label": 1}])

    matrix = build_feature_matrix(pairs, s1, others)
    assert len(matrix) == 1
    assert "name_levenshtein" in matrix.columns
    assert matrix.iloc[0]["label"] == 1


def test_train_model_fits_without_error_on_small_matrix():
    matrix = _df([
        {"name_levenshtein": 1.0, "addr_levenshtein": 1.0, "country_match": 1, "label": 1},
        {"name_levenshtein": 0.1, "addr_levenshtein": 0.1, "country_match": 0, "label": 0},
        {"name_levenshtein": 0.9, "addr_levenshtein": 0.8, "country_match": 1, "label": 1},
        {"name_levenshtein": 0.05, "addr_levenshtein": 0.0, "country_match": 0, "label": 0},
    ])
    model = train_model(matrix, feature_columns=["name_levenshtein", "addr_levenshtein", "country_match"])
    preds = model.predict_proba(matrix[["name_levenshtein", "addr_levenshtein", "country_match"]])
    assert preds.shape == (4, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/train.py
import random

import pandas as pd
from lightgbm import LGBMClassifier

from src.features import build_pair_features
from src.utils_io import parse_id_list


def build_training_pairs(
    s1_df, others_df, ground_truth_df, blocking_candidates,
    max_negatives_per_positive: int = 10, seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    gt_map = {
        s1_id: set(parse_id_list(matched))
        for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"])
    }

    rows = []
    for s1_id, positives in gt_map.items():
        negatives_pool = list(set(blocking_candidates.get(s1_id, set())) - positives)
        cap = max_negatives_per_positive * max(1, len(positives))
        if len(negatives_pool) > cap:
            negatives_pool = rng.sample(negatives_pool, cap)

        for other_id in positives:
            rows.append({"source1_entity_id": s1_id, "other_entity_id": other_id, "label": 1})
        for other_id in negatives_pool:
            rows.append({"source1_entity_id": s1_id, "other_entity_id": other_id, "label": 0})
    return pd.DataFrame(rows, columns=["source1_entity_id", "other_entity_id", "label"])


def build_feature_matrix(pairs_df, s1_df, others_df, embed_lookup=None) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "label"])

    s1_renamed = s1_df.rename(columns={
        "entity_id": "source1_entity_id", "business_name": "name_a",
        "business_address": "addr_a", "country": "country_a",
    })[["source1_entity_id", "name_a", "addr_a", "country_a"]]
    others_renamed = others_df.rename(columns={
        "entity_id": "other_entity_id", "business_name": "name_b",
        "business_address": "addr_b", "country": "country_b",
    })[["other_entity_id", "name_b", "addr_b", "country_b"]]

    merged = pairs_df.merge(s1_renamed, on="source1_entity_id").merge(others_renamed, on="other_entity_id")

    feature_rows = []
    for row in merged.itertuples():
        embed_a = embed_lookup.get(row.source1_entity_id) if embed_lookup else None
        embed_b = embed_lookup.get(row.other_entity_id) if embed_lookup else None
        feats = build_pair_features(
            row.name_a, row.addr_a, row.country_a,
            row.name_b, row.addr_b, row.country_b,
            embed_a=embed_a, embed_b=embed_b,
        )
        feats["source1_entity_id"] = row.source1_entity_id
        feats["other_entity_id"] = row.other_entity_id
        feats["label"] = row.label
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)


def train_model(feature_matrix_df, feature_columns) -> LGBMClassifier:
    model = LGBMClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(feature_matrix_df[feature_columns], feature_matrix_df["label"])
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_train.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/train.py \
        code/business_entity_resolution/tests/test_train.py
git commit -m "feat: add training pair construction, feature matrix, model training"
```

---

## Task 9: Validation split, threshold tuning, model persistence

**Files:**
- Modify: `code/business_entity_resolution/src/train.py` (append)
- Modify: `code/business_entity_resolution/tests/test_train.py` (append)

**Interfaces:**
- Consumes: `macro_f_beta` from `src.metrics`; `parse_id_list` from
  `src.utils_io`; everything already in `src.train` (Task 8)
- Produces:
  - `split_train_validation(s1_df, val_frac=0.2, seed=42) -> tuple[list[str], list[str]]` —
    returns `(train_ids, val_ids)`, S1 `entity_id` values, no overlap,
    `val_ids` sized to `round(len(s1_df) * val_frac)` (minimum 1 if
    `len(s1_df) >= 2`).
  - `tune_threshold(scored_pairs_df, ground_truth_df, thresholds=None) -> float` —
    `scored_pairs_df` has columns `source1_entity_id, other_entity_id, score`;
    tries each candidate threshold (default `numpy.arange(0.1, 1.0, 0.05)`).
    **For each threshold, one vectorized filter + one `groupby` produces every
    entity's prediction set in a single pass** (`scored_pairs_df[score >= threshold].groupby("source1_entity_id")["other_entity_id"].agg(set)`),
    not a per-entity filter of the full table — filtering
    `scored_pairs_df` once per validation entity per threshold is
    `O(thresholds × entities × len(scored_pairs_df))`, which is fine at the
    tiny sizes in this task's unit tests but becomes computationally
    infeasible at real scale (hundreds of thousands of validation entities
    against a scored-pairs table with millions of rows). Scores with
    `macro_f_beta` against the ground-truth dict derived from
    `ground_truth_df` (built via `zip(...)`, not `.iterrows()` — see
    `build_training_pairs` above), returns the threshold with the highest
    score (ties broken by the higher threshold, since higher favors
    precision).
  - `save_model_artifact(model, threshold, feature_columns, path) -> None` —
    `joblib.dump({"model": model, "threshold": threshold, "feature_columns": feature_columns}, path)`.
  - `load_model_artifact(path) -> dict` — `joblib.load(path)`, raises
    `FileNotFoundError` with a clear message if `path` doesn't exist.

- [ ] **Step 1: Write the failing tests**

```python
# appended to code/business_entity_resolution/tests/test_train.py
import os
import pandas as pd
import pytest
from src.train import (
    split_train_validation,
    tune_threshold,
    save_model_artifact,
    load_model_artifact,
    train_model,
)


def test_split_train_validation_no_overlap_and_covers_all_ids():
    s1 = pd.DataFrame({"entity_id": [f"S1-{i:05d}" for i in range(10)]})
    train_ids, val_ids = split_train_validation(s1, val_frac=0.2, seed=42)
    assert set(train_ids) & set(val_ids) == set()
    assert set(train_ids) | set(val_ids) == set(s1["entity_id"])
    assert len(val_ids) == 2


def test_split_train_validation_is_deterministic_for_fixed_seed():
    s1 = pd.DataFrame({"entity_id": [f"S1-{i:05d}" for i in range(10)]})
    a = split_train_validation(s1, val_frac=0.2, seed=42)
    b = split_train_validation(s1, val_frac=0.2, seed=42)
    assert a == b


def test_tune_threshold_picks_threshold_that_maximizes_f_beta():
    # true match S2-00001 scores 0.9, false candidate S2-00002 scores 0.4
    scored = pd.DataFrame([
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "score": 0.9},
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00002", "score": 0.4},
    ])
    gt = pd.DataFrame([{"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001"}])
    best = tune_threshold(scored, gt)
    assert 0.4 < best <= 0.9


def test_save_and_load_model_artifact_roundtrip(tmp_path):
    matrix = pd.DataFrame([
        {"f1": 1.0, "label": 1},
        {"f1": 0.0, "label": 0},
        {"f1": 0.9, "label": 1},
        {"f1": 0.1, "label": 0},
    ])
    model = train_model(matrix, feature_columns=["f1"])
    path = str(tmp_path / "model.joblib")
    save_model_artifact(model, threshold=0.55, feature_columns=["f1"], path=path)
    artifact = load_model_artifact(path)
    assert artifact["threshold"] == 0.55
    assert artifact["feature_columns"] == ["f1"]
    assert hasattr(artifact["model"], "predict_proba")


def test_load_model_artifact_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "no_model.joblib"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_model_artifact(str(missing))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'split_train_validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# appended to code/business_entity_resolution/src/train.py
import os
import random

import joblib
import numpy as np

from src.metrics import macro_f_beta
from src.utils_io import parse_id_list


def split_train_validation(s1_df, val_frac: float = 0.2, seed: int = 42):
    ids = list(s1_df["entity_id"])
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    val_size = max(1, round(len(ids) * val_frac)) if len(ids) >= 2 else 0
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return train_ids, val_ids


def tune_threshold(scored_pairs_df, ground_truth_df, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    gt_map = {
        s1_id: set(parse_id_list(matched))
        for s1_id, matched in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"])
    }

    best_threshold = thresholds[0]
    best_score = -1.0
    for threshold in thresholds:
        above = scored_pairs_df[scored_pairs_df["score"] >= threshold]
        grouped = above.groupby("source1_entity_id")["other_entity_id"].agg(set).to_dict()
        predictions = {s1_id: grouped.get(s1_id, set()) for s1_id in gt_map}
        score = macro_f_beta(predictions, gt_map)
        if score >= best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def save_model_artifact(model, threshold: float, feature_columns: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": feature_columns}, path)


def load_model_artifact(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model artifact found at {path}. Run train.py first.")
    return joblib.load(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_train.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/train.py \
        code/business_entity_resolution/tests/test_train.py
git commit -m "feat: add validation split, threshold tuning, model persistence"
```

---

## Task 10: End-to-end inference pipeline

**Files:**
- Create: `code/business_entity_resolution/src/infer.py`
- Create: `code/business_entity_resolution/tests/test_infer.py`

**Interfaces:**
- Consumes: `read_source_tsv`, `write_id_mapping_tsv`, `parse_id_list` from
  `src.utils_io`; `token_overlap_candidates`, `address_prefix_candidates`,
  `embedding_knn_candidates`, `union_candidates`, `candidates_to_rows`,
  `default_embedder` from `src.blocking`; `build_pair_features` from
  `src.features`; `load_model_artifact` from `src.train`
- Produces:
  - `generate_candidates(s1_df, s2_df, s3_df, embedder=None) -> dict[str, set[str]]` —
    concatenates `s2_df`/`s3_df` into one `other_df`, runs
    `token_overlap_candidates`, `address_prefix_candidates`, and (only when
    `embedder` is not `None`) `embedding_knn_candidates`, unions the
    results. This is the function whose output becomes `candidate_pairs.tsv`.
  - `score_candidates(s1_df, others_df, candidates, model, feature_columns) -> pandas.DataFrame` —
    columns `source1_entity_id, other_entity_id, score`, one row per
    `(s1_id, other_id)` in `candidates`. Flattens `candidates` into a pairs
    DataFrame, **merges** it against `s1_df`/`others_df` (vectorized pandas
    merge, not a `.loc` lookup per pair — at millions of candidate pairs a
    lookup per pair is measurably slower), then builds features via
    `itertuples()` over the merged frame (same pattern as
    `train.build_feature_matrix`) before a single batched `model.predict_proba` call.
  - `assign_matches(scored_df, threshold) -> dict[str, list[str]]` — for
    every `source1_entity_id` present in `scored_df`, list of `other_entity_id`
    with `score >= threshold`; entities with none get `[]` (never omitted
    by this function — omission-safety is enforced by `run_inference`
    iterating over every S1 id explicitly).
  - `run_inference(dataset_dir: str, model_path: str, output_dir: str, use_embeddings: bool = True) -> None` —
    orchestrates the full pipeline: `read_source_tsv` on
    `{dataset_dir}/test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`
    (fail-fast per `utils_io`'s existing behavior); `generate_candidates`;
    writes `{output_dir}/candidate_pairs.tsv` via `write_id_mapping_tsv`;
    `load_model_artifact(model_path)`; `score_candidates`; `assign_matches`;
    ensures every `entity_id` in `test_source1.tsv` has a key in the result
    dict (defaulting to `[]`) before writing
    `{output_dir}/matching_results.tsv`.

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_infer.py
import pandas as pd
import pytest

from src.infer import generate_candidates, score_candidates, assign_matches, run_inference
from src.train import train_model


def _df(rows):
    return pd.DataFrame(rows)


def test_generate_candidates_unions_token_and_address_strategies_without_embedder():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme Traders", "business_address": "123 Main St", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme Traders Inc", "business_address": "999 Other Rd", "country": "US"}])
    s3 = _df([{"entity_id": "S3-00001", "business_name": "Nothing Alike", "business_address": "123 Main Street", "country": "US"}])

    result = generate_candidates(s1, s2, s3, embedder=None)
    assert result["S1-00001"] == {"S2-00001", "S3-00001"}


def test_generate_candidates_never_returns_source1_ids():
    s1 = _df([{"entity_id": "S1-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s2 = _df([{"entity_id": "S2-00001", "business_name": "Acme", "business_address": "", "country": "US"}])
    s3 = _df([{"entity_id": "S3-00001", "business_name": "Acme", "business_address": "", "country": "US"}])

    result = generate_candidates(s1, s2, s3, embedder=None)
    assert all(cid.startswith(("S2-", "S3-")) for cid in result["S1-00001"])


def test_assign_matches_gives_empty_list_for_entity_with_no_scores_above_threshold():
    scored = _df([{"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "score": 0.2}])
    result = assign_matches(scored, threshold=0.5)
    assert result["S1-00001"] == []


def test_assign_matches_keeps_scores_above_threshold():
    scored = _df([
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00001", "score": 0.9},
        {"source1_entity_id": "S1-00001", "other_entity_id": "S2-00002", "score": 0.1},
    ])
    result = assign_matches(scored, threshold=0.5)
    assert result["S1-00001"] == ["S2-00001"]


def test_run_inference_writes_row_for_every_source1_entity_including_singletons(tmp_path):
    dataset_dir = tmp_path / "test"
    dataset_dir.mkdir()
    (dataset_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Traders\t123 Main St\tUS\n"
        "S1-00002\tCompletely Unique Business\t999 Nowhere Ave\tFrance\n"
    )
    (dataset_dir / "test_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00001\tAcme Traders Inc\t123 Main Street\tUS\n"
    )
    (dataset_dir / "test_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    )

    matrix = _df([
        {"name_levenshtein": 1.0, "label": 1},
        {"name_levenshtein": 0.0, "label": 0},
    ])
    model = train_model(matrix, feature_columns=["name_levenshtein"])
    model_path = tmp_path / "model.joblib"
    from src.train import save_model_artifact
    save_model_artifact(model, threshold=0.5, feature_columns=["name_levenshtein"], path=str(model_path))

    output_dir = tmp_path / "output"
    run_inference(str(dataset_dir), str(model_path), str(output_dir), use_embeddings=False)

    results_text = (output_dir / "matching_results.tsv").read_text()
    assert "S1-00001" in results_text
    assert "S1-00002\t" in results_text  # singleton still gets a row

    candidates_text = (output_dir / "candidate_pairs.tsv").read_text()
    assert "S1-00001" in candidates_text


def test_run_inference_missing_dataset_file_fails_fast(tmp_path):
    dataset_dir = tmp_path / "test"
    dataset_dir.mkdir()
    output_dir = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        run_inference(str(dataset_dir), str(tmp_path / "model.joblib"), str(output_dir), use_embeddings=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.infer'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/infer.py
import os

import pandas as pd

from src.blocking import (
    token_overlap_candidates,
    address_prefix_candidates,
    embedding_knn_candidates,
    union_candidates,
    candidates_to_rows,
    default_embedder,
)
from src.features import build_pair_features
from src.train import load_model_artifact
from src.utils_io import read_source_tsv, write_id_mapping_tsv


def generate_candidates(s1_df, s2_df, s3_df, embedder=None) -> dict:
    others_df = pd.concat([s2_df, s3_df], ignore_index=True)
    strategies = [
        token_overlap_candidates(s1_df, others_df),
        address_prefix_candidates(s1_df, others_df),
    ]
    if embedder is not None:
        strategies.append(embedding_knn_candidates(s1_df, others_df, embedder=embedder))
    return union_candidates(*strategies)


def score_candidates(s1_df, others_df, candidates: dict, model, feature_columns) -> pd.DataFrame:
    pairs = pd.DataFrame(
        [(s1_id, other_id) for s1_id, other_ids in candidates.items() for other_id in other_ids],
        columns=["source1_entity_id", "other_entity_id"],
    )
    if pairs.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "score"])

    s1_renamed = s1_df.rename(columns={
        "entity_id": "source1_entity_id", "business_name": "name_a",
        "business_address": "addr_a", "country": "country_a",
    })[["source1_entity_id", "name_a", "addr_a", "country_a"]]
    others_renamed = others_df.rename(columns={
        "entity_id": "other_entity_id", "business_name": "name_b",
        "business_address": "addr_b", "country": "country_b",
    })[["other_entity_id", "name_b", "addr_b", "country_b"]]

    merged = pairs.merge(s1_renamed, on="source1_entity_id").merge(others_renamed, on="other_entity_id")

    feature_rows = []
    for row in merged.itertuples():
        feats = build_pair_features(
            row.name_a, row.addr_a, row.country_a,
            row.name_b, row.addr_b, row.country_b,
        )
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    scores = model.predict_proba(feat_df[feature_columns])[:, 1]
    return pd.DataFrame({
        "source1_entity_id": merged["source1_entity_id"].values,
        "other_entity_id": merged["other_entity_id"].values,
        "score": scores,
    })


def assign_matches(scored_df: pd.DataFrame, threshold: float) -> dict:
    result = {}
    for s1_id, group in scored_df.groupby("source1_entity_id"):
        matched = group[group["score"] >= threshold]["other_entity_id"].tolist()
        result[s1_id] = matched
    return result


def run_inference(dataset_dir: str, model_path: str, output_dir: str, use_embeddings: bool = True) -> None:
    s1_df = read_source_tsv(os.path.join(dataset_dir, "test_source1.tsv"))
    s2_df = read_source_tsv(os.path.join(dataset_dir, "test_source2.tsv"))
    s3_df = read_source_tsv(os.path.join(dataset_dir, "test_source3.tsv"))
    others_df = pd.concat([s2_df, s3_df], ignore_index=True)

    embedder = default_embedder if use_embeddings else None
    candidates = generate_candidates(s1_df, s2_df, s3_df, embedder=embedder)

    candidate_rows = candidates_to_rows(candidates)
    for entity_id in s1_df["entity_id"]:
        candidate_rows.setdefault(entity_id, [])
    write_id_mapping_tsv(
        candidate_rows, os.path.join(output_dir, "candidate_pairs.tsv"),
        id_col="source1_entity_id", list_col="candidate_entity_ids",
    )

    artifact = load_model_artifact(model_path)
    scored_df = score_candidates(s1_df, others_df, candidates, artifact["model"], artifact["feature_columns"])
    matches = assign_matches(scored_df, artifact["threshold"])

    for entity_id in s1_df["entity_id"]:
        matches.setdefault(entity_id, [])
    write_id_mapping_tsv(
        matches, os.path.join(output_dir, "matching_results.tsv"),
        id_col="source1_entity_id", list_col="matched_entity_ids",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_infer.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/infer.py \
        code/business_entity_resolution/tests/test_infer.py
git commit -m "feat: add end-to-end inference pipeline with fail-fast dataset checks"
```

---

## Task 11: validate_submission.py wiring, README, full test suite pass

**Files:**
- Modify: `code/business_entity_resolution/src/infer.py` (append)
- Create: `code/business_entity_resolution/tests/test_infer_validation_hook.py`
- Create: `code/business_entity_resolution/README.md`

**Interfaces:**
- Consumes: `run_inference` from this module (already defined, Task 10)
- Produces:
  - `maybe_run_validator(output_dir: str, test_dir: str, validator_path: str) -> str | None` —
    if `validator_path` exists, runs it via `subprocess.run` with
    `--matching {output_dir}/matching_results.tsv --candidate {output_dir}/candidate_pairs.tsv --test-dir {test_dir}`,
    returns its captured stdout; if `validator_path` does not exist, returns
    `None` without raising (validator isn't in the repo yet per the spec's
    open items).

- [ ] **Step 1: Write the failing tests**

```python
# code/business_entity_resolution/tests/test_infer_validation_hook.py
from src.infer import maybe_run_validator


def test_maybe_run_validator_returns_none_when_script_absent(tmp_path):
    result = maybe_run_validator(
        output_dir=str(tmp_path / "output"),
        test_dir=str(tmp_path / "dataset" / "test"),
        validator_path=str(tmp_path / "utils" / "validate_submission.py"),
    )
    assert result is None


def test_maybe_run_validator_runs_script_when_present(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    validator_path = tmp_path / "utils" / "validate_submission.py"
    validator_path.parent.mkdir(parents=True)
    validator_path.write_text(
        "import sys\n"
        "print('PASS')\n"
        "sys.exit(0)\n"
    )
    result = maybe_run_validator(str(output_dir), str(test_dir), str(validator_path))
    assert result is not None
    assert "PASS" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_infer_validation_hook.py -v`
Expected: FAIL with `ImportError: cannot import name 'maybe_run_validator'`

- [ ] **Step 3: Write minimal implementation**

```python
# appended to code/business_entity_resolution/src/infer.py
import subprocess
import sys


def maybe_run_validator(output_dir: str, test_dir: str, validator_path: str):
    if not os.path.exists(validator_path):
        return None
    result = subprocess.run(
        [
            sys.executable, validator_path,
            "--matching", os.path.join(output_dir, "matching_results.tsv"),
            "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
            "--test-dir", test_dir,
        ],
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr
```

Also wire it into the end of `run_inference` (edit the existing function body
in `src/infer.py`, replacing its final lines):

```python
    write_id_mapping_tsv(
        matches, os.path.join(output_dir, "matching_results.tsv"),
        id_col="source1_entity_id", list_col="matched_entity_ids",
    )

    validator_output = maybe_run_validator(
        output_dir, dataset_dir, os.path.join(os.path.dirname(dataset_dir), "..", "utils", "validate_submission.py")
    )
    if validator_output:
        print(validator_output)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/business_entity_resolution && python -m pytest tests/test_infer_validation_hook.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write README.md**

```markdown
# code/business_entity_resolution/README.md
# Kinfolk — Business Entity Resolution Pipeline

## Setup

    pip install -r requirements.txt

## Data layout

The challenge dataset is under `student_resource/` at the repo root
(unzipped from the portal's `student_resource.zip`):

    student_resource/dataset/train/train_source1.tsv
    student_resource/dataset/train/train_source2.tsv
    student_resource/dataset/train/train_source3.tsv
    student_resource/dataset/train/train_ground_truth.tsv
    student_resource/dataset/test/test_source1.tsv
    student_resource/dataset/test/test_source2.tsv
    student_resource/dataset/test/test_source3.tsv
    student_resource/utils/validate_submission.py
    student_resource/Documentation_template.md

Real sizes: 2.2M-5.3M rows per source file, ~2.5GB total. `train_cli`/
`infer_cli` (see Follow-up in the plan) take `--dataset-dir` as an argument,
so they work against this path without hardcoding it.

## Reproduce: data -> blocking -> matching -> output

1. Train the model (reads `student_resource/dataset/train/*`, writes
   `model.joblib` at repo root):

       python -m src.train_cli --dataset-dir student_resource/dataset/train --model-path model.joblib

2. Run inference (reads `student_resource/dataset/test/*` and
   `model.joblib`, writes both output TSVs under `output/`):

       python -m src.infer_cli --dataset-dir student_resource/dataset/test --model-path model.joblib --output-dir output

3. Validate (`infer_cli` runs this automatically and prints the result,
   since `student_resource/utils/validate_submission.py` is present; to run
   it manually):

       python3 student_resource/utils/validate_submission.py \
         --matching output/matching_results.tsv \
         --candidate output/candidate_pairs.tsv \
         --test-dir student_resource/dataset/test

## Tests

    python -m pytest tests/ -v
```

- [ ] **Step 6: Run the full test suite**

Run: `cd code/business_entity_resolution && python -m pytest tests/ -v`
Expected: PASS (all tests across all modules)

- [ ] **Step 7: Commit**

```bash
git add code/business_entity_resolution/src/infer.py \
        code/business_entity_resolution/tests/test_infer_validation_hook.py \
        code/business_entity_resolution/README.md
git commit -m "feat: wire optional validate_submission.py hook and add README"
```

---

## Follow-up (not in this plan — needs the real dataset run, not just fixtures)

The dataset, `validate_submission.py`, and `Documentation_template.md` are
already in the repo under `student_resource/` (see Task 11's README). What's
still deferred past this plan is exercising the pipeline against that real,
multi-million-row data — this plan's tasks are scoped to building each
module correctly (verified via small fixtures), not to a full real-data
run, which is its own significant time cost (embedding ~14M records,
training on millions of pairs) better scheduled deliberately:
1. Add thin CLI entrypoints `src/train_cli.py` / `src/infer_cli.py` (argparse
   wrappers around `train.py`/`infer.py` functions — referenced in the
   README above but not yet created, since they need real file paths to
   smoke-test against).
2. Run `train.py`'s functions end-to-end against real training data, inspect
   the validation F_0.5, adjust blocking recall/threshold/negative-sampling-cap
   as needed. Watch memory (9.5GB available) and wall-clock time; if the
   FAISS IVF partition-training step or the itertuples feature loop proves
   too slow at real volumes, that's the next thing to profile and fix.
3. Run `infer.py` against real test data, confirm `validate_submission.py`
   passes.
4. Fill in `Documentation_template.md` with the methodology described in the
   spec.
5. Assemble the final submission zip per the problem statement's structure
   (`output/`, `code/business_entity_resolution/`, the filled methodology
   doc — see the original PDF's "Final Submission Package" section).
