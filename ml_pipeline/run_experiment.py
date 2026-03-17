"""
run_experiment.py
Main script for batch execution of the entire experiment pipeline.

Execution order:
  1) Generate synthetic dataset (when real data is unavailable)
  2) Train and compare 3 models
  3) Hybrid ensemble evaluation
  4) Ablation Study
  5) Statistical significance testing

Usage:
  python run_experiment.py                        # Use synthetic data
  python run_experiment.py dataset/real_data.csv  # Use real data
"""

import sys
import os

# Execute relative to the ml_pipeline directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_dataset import generate_dataset
from model_trainer import train_and_evaluate_all
from evaluate import (
    run_full_evaluation,
    ablation_study,
    evaluate_hybrid_ensemble,
    statistical_validation,
)


def main():
    csv_path = 'dataset/crypto_features.csv'

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            sys.exit(1)
        print(f"Using real data: {csv_path}")
    else:
        if not os.path.exists(csv_path):
            print("=" * 60)
            print("  [Step 0] Generating synthetic dataset")
            print("=" * 60)
            generate_dataset(csv_path)
        else:
            print(f"Using existing dataset: {csv_path}")

    # ── Step 1: Train and compare 3 models (Paper Table 3) ──
    print("\n" + "=" * 60)
    print("  [Step 1] Model training and performance comparison")
    print("=" * 60)
    results, X_train, X_val, X_test, y_train, y_val, y_test, le = \
        train_and_evaluate_all(csv_path, save_dir='models')

    # ── Step 2: Full evaluation (ensemble + ablation + t-test) ──
    print("\n" + "=" * 60)
    print("  [Step 2] Running full evaluation")
    print("=" * 60)
    run_full_evaluation(csv_path)

    print("\n" + "=" * 60)
    print("  All experiments completed")
    print("=" * 60)
    print("\n  Generated files:")
    print("    models/xgb_classifier.pkl   - XGBoost model")
    print("    models/rf_classifier.pkl    - Random Forest model")
    print("    models/mlp_classifier.pkl   - MLP model")
    print("    models/mlp_scaler.pkl       - StandardScaler for MLP")
    print("    models/label_encoder.pkl    - Label encoder")
    print(f"    {csv_path}    - Dataset")


if __name__ == '__main__':
    main()
