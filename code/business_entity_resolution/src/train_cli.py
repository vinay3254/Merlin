import argparse

from src.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Train the entity-resolution matching model.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv")
    parser.add_argument("--model-path", required=True, help="Where to save the trained model artifact")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true", default=False, help="Train model using GPU")
    parser.add_argument("--max-train-entities", type=int, default=None, help="Sample max S1 entities for training")
    parser.add_argument("--use-embeddings", action="store_true", default=False, help="Enable embedding-kNN blocking (GPU-accelerated; adds significant runtime)")
    parser.add_argument("--candidates-cache", type=str, default=None, help="Path to cache/reuse computed blocking candidates. Skips the ~2hr blocking pass on a cache hit -- use to iterate on classifier/feature/negative-sampling changes quickly.")
    parser.add_argument("--max-negatives-per-positive", type=int, default=10, help="Cap on sampled negative pairs per positive, per S1 entity")
    args = parser.parse_args()

    summary = run_training(
        args.dataset_dir,
        args.model_path,
        val_frac=args.val_frac,
        seed=args.seed,
        use_gpu=args.gpu,
        max_train_entities=args.max_train_entities,
        use_embeddings=args.use_embeddings,
        candidates_cache_path=args.candidates_cache,
        max_negatives_per_positive=args.max_negatives_per_positive,
    )
    print(f"Trained model saved to {args.model_path}")
    print(f"Validation F_0.5: {summary['val_f_beta']:.4f}")
    print(f"Chosen threshold: {summary['threshold']:.2f}")
    print(f"Training pairs: {summary['n_train_pairs']}, validation entities: {summary['n_val_entities']}")


if __name__ == "__main__":
    main()
