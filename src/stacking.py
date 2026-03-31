# ===============================
# NEW FILE: src/stacking.py
# ===============================
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

from config import Config

def run_stacking():
    """Run stacking ensemble using pre-trained models' OOF predictions"""

    print("Loading OOF predictions...")
    
    # Check if OOF files exist
    oof_dir = "data/processed/oof/"
    if not os.path.exists(f"{oof_dir}/lgb_oof.npy"):
        raise FileNotFoundError("LightGBM OOF predictions not found. Run train.py first.")
    if not os.path.exists(f"{oof_dir}/cb_oof.npy"):
        raise FileNotFoundError("CatBoost OOF predictions not found. Run train_cb.py first.")

    # Load OOF predictions
    lgb_oof = np.load(f"{oof_dir}/lgb_oof.npy")
    cb_oof = np.load(f"{oof_dir}/cb_oof.npy")

    lgb_test = np.load(f"{oof_dir}/lgb_test.npy")
    cb_test = np.load(f"{oof_dir}/cb_test.npy")

    # Stack features (each model prediction becomes a feature)
    X_stack = np.column_stack([lgb_oof, cb_oof])
    X_test_stack = np.column_stack([lgb_test, cb_test])

    print(f"Stacking features shape: {X_stack.shape}")

    # Load target
    train = pd.read_csv("data/raw/train.csv")
    y = train[Config.TARGET]

    # Train meta model with cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
    meta_oof = np.zeros(len(X_stack))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_stack, y)):
        meta_model = LogisticRegression(random_state=Config.RANDOM_STATE)
        meta_model.fit(X_stack[tr_idx], y.iloc[tr_idx])
        meta_oof[val_idx] = meta_model.predict_proba(X_stack[val_idx])[:, 1]

    # Final meta model on full data
    final_meta_model = LogisticRegression(random_state=Config.RANDOM_STATE)
    final_meta_model.fit(X_stack, y)

    # Generate final predictions
    final_preds = final_meta_model.predict_proba(X_test_stack)[:, 1]

    # Calculate stacking CV score
    stacking_score = roc_auc_score(y, meta_oof)
    print(f"Stacking CV Score: {stacking_score:.5f}")

    # Save stacked submission
    test = pd.read_csv("data/raw/test.csv")
    submission = pd.DataFrame({
        "Id": test.index,
        Config.TARGET: final_preds
    })

    submission.to_csv("data/submissions/stacked_submission.csv", index=False)
    print("Stacked submission saved to data/submissions/stacked_submission.csv")

    return stacking_score

def run_weighted_ensemble(weights=None):
    """Simple weighted ensemble without meta-learning"""
    
    if weights is None:
        weights = [0.5, 0.5]  # Equal weights by default

    oof_dir = "data/processed/oof/"
    
    # Load OOF predictions
    lgb_oof = np.load(f"{oof_dir}/lgb_oof.npy")
    cb_oof = np.load(f"{oof_dir}/cb_oof.npy")

    lgb_test = np.load(f"{oof_dir}/lgb_test.npy")
    cb_test = np.load(f"{oof_dir}/cb_test.npy")

    # Weighted ensemble
    ensemble_oof = weights[0] * lgb_oof + weights[1] * cb_oof
    ensemble_test = weights[0] * lgb_test + weights[1] * cb_test

    # Calculate ensemble score
    train = pd.read_csv("data/raw/train.csv")
    y = train[Config.TARGET]
    
    ensemble_score = roc_auc_score(y, ensemble_oof)
    print(f"Weighted Ensemble CV Score: {ensemble_score:.5f}")

    # Save weighted ensemble submission
    test = pd.read_csv("data/raw/test.csv")
    submission = pd.DataFrame({
        "Id": test.index,
        Config.TARGET: ensemble_test
    })

    submission.to_csv("data/submissions/weighted_ensemble_submission.csv", index=False)
    print("Weighted ensemble submission saved to data/submissions/weighted_ensemble_submission.csv")

    return ensemble_score

if __name__ == "__main__":
    print("Running stacking ensemble...")
    stacking_score = run_stacking()
    
    print("\nRunning weighted ensemble...")
    weighted_score = run_weighted_ensemble()
    
    print(f"\nResults Summary:")
    print(f"Stacking CV Score: {stacking_score:.5f}")
    print(f"Weighted Ensemble CV Score: {weighted_score:.5f}")