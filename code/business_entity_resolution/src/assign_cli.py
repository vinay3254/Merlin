"""
Build matching_results.tsv from a scores TSV (written by src.rescore_cli)
using assign_matches' one-S1-per-record rule, at one or more thresholds.

With several --thresholds, prints match-count stats for each (for choosing
a threshold) and writes a matching_results file per threshold into
--output-dir as matching_results_t<threshold>.tsv.
"""
import argparse
import os
import sys

import pandas as pd

from src.infer import assign_matches
from src.utils_io import read_source_tsv, write_id_mapping_tsv


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--source1", required=True, help="test_source1.tsv, so every S1 entity gets a row")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    args = parser.parse_args(argv)

    scored = pd.read_csv(
        args.scores, sep="\t", keep_default_na=False,
        dtype={"source1_entity_id": "category", "other_entity_id": "category", "score": "float32"},
    )
    s1_ids = read_source_tsv(args.source1)["entity_id"]
    os.makedirs(args.output_dir, exist_ok=True)

    for threshold in args.thresholds:
        matches = assign_matches(scored, threshold)
        for entity_id in s1_ids:
            matches.setdefault(entity_id, [])
        sizes = pd.Series([len(v) for v in matches.values()])
        n_above = int((scored["score"] >= threshold).sum())
        print(
            f"threshold={threshold:.2f} pairs_above={n_above} assigned={int(sizes.sum())} "
            f"dropped_as_duplicate={n_above - int(sizes.sum())} mean_matches={sizes.mean():.3f} "
            f"empty_frac={(sizes == 0).mean():.4f}",
            flush=True,
        )
        write_id_mapping_tsv(
            matches, os.path.join(args.output_dir, f"matching_results_t{threshold:.2f}.tsv"),
            id_col="source1_entity_id", list_col="matched_entity_ids",
        )


if __name__ == "__main__":
    sys.exit(main())
