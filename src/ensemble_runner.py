# ===============================
# ensemble_runner.py - SIMPLIFIED
# ===============================
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Import training functions (these use W&B internally)
from train import run_training
from stacking import run_stacking, run_weighted_ensemble
from optuna_tuning import run_hyperparameter_tuning
from config import Config

# Global results dictionary
results = {}


def train_single_model(model_name):
    """Train a single model and return CV score"""
    print(f"\n{'='*30}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*30}")
    
    if model_name == "lightgbm":
        score = run_training("lightgbm")
    elif model_name == "catboost":
        score = run_training("catboost")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    results[model_name] = score
    print(f"{model_name.upper()} CV Score: {score:.5f}")
    return score


def train_all_base_models():
    """Train all base models"""
    print("\n🚀 Training all base models...")
    
    # Ensure OOF directory exists
    oof_dir = Path("data/processed/oof/")
    oof_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    lgb_score = train_single_model("lightgbm")
    cb_score = train_single_model("catboost")
    
    return lgb_score, cb_score


def create_ensembles():
    """Create ensemble predictions"""
    print(f"\n{'='*30}")
    print("Creating Ensembles")
    print(f"{'='*30}")
    
    oof_dir = Path("data/processed/oof/")
    
    # Check if OOF files exist
    if not (oof_dir / "lgb_oof.npy").exists():
        print("❌ LightGBM OOF not found. Run base models first.")
        return None, None
    if not (oof_dir / "cb_oof.npy").exists():
        print("❌ CatBoost OOF not found. Run base models first.")
        return None, None
    
    # Stacking
    print("🔗 Running Stacking...")
    stacking_score = run_stacking()
    results["stacking"] = stacking_score
    
    # Weighted ensemble
    print("⚖️ Running Weighted Ensemble...")
    weighted_score = run_weighted_ensemble()
    results["weighted_ensemble"] = weighted_score
    
    return stacking_score, weighted_score


def optimize_hyperparameters(model="lightgbm", n_trials=50):
    """Run hyperparameter optimization"""
    print(f"\n{'='*40}")
    print(f"Optimizing {model.upper()} Hyperparameters")
    print(f"{'='*40}")
    
    from features import build_features
    
    # Load data
    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")
    X, y, X_test = build_features(train_df, test_df)
    
    # Optimize
    if model == "lightgbm":
        best_params = run_hyperparameter_tuning(X, y, n_trials=n_trials)
        print("\n📋 Best LightGBM parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        return best_params
    
    return None


def print_results_summary():
    """Print results summary"""
    print(f"\n{'='*40}")
    print("🏆 RESULTS SUMMARY")
    print(f"{'='*40}")
    
    if not results:
        print("No results available.")
        return
    
    # Sort by score (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<20} {'CV Score':<10}")
    print("-" * 32)
    
    for model, score in sorted_results:
        print(f"{model:<20} {score:<10.5f}")
    
    print(f"\n🥇 Best: {sorted_results[0][0]} ({sorted_results[0][1]:.5f})")
    
    # Show submission files
    sub_dir = Path("data/submissions/")
    if sub_dir.exists():
        print(f"\n📁 Submission files:")
        for f in sub_dir.glob("*.csv"):
            print(f"  - {f.name}")


def compare_submissions():
    """Compare submission predictions"""
    sub_dir = Path("data/submissions/")
    if not sub_dir.exists():
        return
    
    submissions = {}
    for sub_file in sub_dir.glob("*.csv"):
        df = pd.read_csv(sub_file)
        submissions[sub_file.stem] = df[Config.TARGET].values
    
    if len(submissions) >= 2:
        print(f"\n📊 Submission Correlations:")
        corr_matrix = pd.DataFrame(submissions).corr()
        print(corr_matrix.round(4))


def run_full_pipeline(optimize_first=False, n_trials=50, use_wandb=True):
    """
    Run complete pipeline
    
    Parameters:
    - optimize_first: Run hyperparameter optimization first
    - n_trials: Number of Optuna trials
    - use_wandb: Whether to use Weights & Biases (the training functions use it)
    """
    print("🎯 Full Kaggle Pipeline")
    print("=" * 40)
    
    # Step 1: Hyperparameter optimization (optional)
    if optimize_first:
        print("Step 1: Hyperparameter Optimization")
        best_params = optimize_hyperparameters(n_trials=n_trials)
        print("⚠️  Update src/config.py with best parameters!")
    
    # Step 2: Train base models (uses W&B internally)
    print("\nStep 2: Training Base Models")
    if not use_wandb:
        print("⚠️  Note: Individual training functions still use W&B internally")
    lgb_score, cb_score = train_all_base_models()
    
    # Step 3: Create ensembles
    print("\nStep 3: Creating Ensembles")
    stacking_score, weighted_score = create_ensembles()
    
    # Step 4: Summary
    print_results_summary()
    compare_submissions()
    
    return results


# Simple CLI interface
def main():
    """Simple main function"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            train_all_base_models()
        elif command == "ensemble":
            create_ensembles()
        elif command == "optimize":
            n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            optimize_hyperparameters(n_trials=n_trials)
        elif command == "full":
            run_full_pipeline()
        else:
            print("Usage: python ensemble_runner.py [train|ensemble|optimize|full]")
    else:
        # Default: run full pipeline
        run_full_pipeline()


if __name__ == "__main__":
    main()