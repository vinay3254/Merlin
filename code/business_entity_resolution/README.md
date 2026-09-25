# Kinfolk — Business Entity Resolution Pipeline

## Setup

Run from `code/business_entity_resolution/`:

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

Real sizes: 2.2M-5.3M rows per source file, ~2.5GB total.

## Reproduce: data -> blocking -> matching -> output

**Note:** The commands below show the intended pipeline interface using thin CLI
wrappers (`src/train_cli.py` and `src/infer_cli.py`). These wrappers are not yet
built—they are tracked as follow-up work (see the implementation plan's Follow-up
section). Today, the underlying functions in `src/train.py` and `src/infer.py` are
usable directly from Python, and the full pipeline is verified via the test suite
(`pytest tests/`).

Once those CLI wrappers are added, you will be able to run from the **repo root**:

1. Train the model (reads `student_resource/dataset/train/*`, writes
   `model.joblib` at repo root):

       python -m src.train_cli --dataset-dir student_resource/dataset/train --model-path model.joblib

2. Run inference (reads `student_resource/dataset/test/*` and
   `model.joblib`, writes both output TSVs under `output/`):

       python -m src.infer_cli --dataset-dir student_resource/dataset/test --model-path model.joblib --output-dir output

3. Validate (`infer_cli` runs this automatically and prints the result,
   since `student_resource/utils/validate_submission.py` is present; to run
   it manually from **repo root**):

       python3 student_resource/utils/validate_submission.py \
         --matching output/matching_results.tsv \
         --candidate output/candidate_pairs.tsv \
         --test-dir student_resource/dataset/test

## Tests

Run from `code/business_entity_resolution/`:

    python -m pytest tests/ -v
