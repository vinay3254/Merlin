import argparse

from src.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Train the entity-resolution matching model.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv")
    parser.add_argument("--model-path", required=True, help="Where to save the trained model artifact")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_training(args.dataset_dir, args.model_path, val_frac=args.val_frac, seed=args.seed)
    print(f"Trained model saved to {args.model_path}")
    print(f"Validation F_0.5: {summary['val_f_beta']:.4f}")
    print(f"Chosen threshold: {summary['threshold']:.2f}")
    print(f"Training pairs: {summary['n_train_pairs']}, validation entities: {summary['n_val_entities']}")


if __name__ == "__main__":
    main()
