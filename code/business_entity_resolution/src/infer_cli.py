import argparse

from src.infer import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run inference: generate matches for the test set.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    parser.add_argument("--model-path", required=True, help="Path to a saved model artifact")
    parser.add_argument("--output-dir", required=True, help="Where to write matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--use-embeddings", action="store_true", help="Enable embedding-based blocking (default off — deferred, real-scale memory issues; requires sentence-transformers)")
    args = parser.parse_args()

    run_inference(args.dataset_dir, args.model_path, args.output_dir, use_embeddings=args.use_embeddings)
    print(f"Wrote {args.output_dir}/matching_results.tsv and {args.output_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
