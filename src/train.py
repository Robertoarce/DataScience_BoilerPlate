# ===============================
# train.py - UNIFIED TRAINING
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import wandb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os

from config import Config
from features import build_features
from utils import plot_feature_importance, save_model, load_model

def get_model(model_type="lightgbm"):
    """Get model based on type"""
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(**Config.LGB_PARAMS)
    elif model_type == "catboost":
        return CatBoostClassifier(**Config.CB_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_training(model_type="lightgbm"):
    """
    Unified training pipeline
    
    Args:
        model_type: "lightgbm" or "catboost"
    """
    
    # W&B project name based on model
    project_name = f"kaggle-{model_type}"
    model_config = Config.LGB_PARAMS if model_type == "lightgbm" else Config.CB_PARAMS
    
    wandb.init(project=project_name, config=model_config)

    print(f"🚀 Training {model_type.upper()} model...")
    
    # Load data
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    X, y, X_test = build_features(train, test)

    # Cross-validation setup
    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS,
        shuffle=True,
        random_state=Config.RANDOM_STATE
    )

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    feature_importance = []
    models = []  # Store models for feature importance

    # Training loop
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = get_model(model_type)

        # Model-specific training
        if model_type == "lightgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        else:  # catboost
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        preds += model.predict_proba(X_test)[:, 1] / Config.N_SPLITS

        # Store model and feature importance
        models.append(model)
        if model_type == "lightgbm":
            feature_importance.append(model.feature_importances_)
        elif model_type == "catboost":
            feature_importance.append(model.get_feature_importance())

        score = roc_auc_score(y_val, oof[val_idx])
        wandb.log({f"fold_{fold}": score})
        print(f"Fold {fold}: {score:.5f}")

    final_score = roc_auc_score(y, oof)
    wandb.log({"oof_score": final_score})

    # Feature importance analysis
    print("📊 Analyzing feature importance...")
    avg_importance = np.mean(feature_importance, axis=0)
    
    # Get feature names (try to get from preprocessor if possible)
    try:
        # This might not work depending on sklearn version and preprocessing
        feature_names = None  # Would need to get from preprocessor
    except:
        feature_names = None
    
    # Plot and save feature importance
    feat_imp_df = plot_feature_importance(models[-1], feature_names, model_type)
    
    # Save feature importance to CSV
    feat_imp_df.to_csv(f"plots/{model_type}_feature_importance.csv", index=False)
    
    # Log feature importance to W&B
    wandb.log({"feature_importance": wandb.Table(dataframe=feat_imp_df.head(20))})

    # Save final model (last fold)
    save_model(models[-1], model_type)

    # Save OOF predictions
    oof_dir = "data/processed/oof/"
    os.makedirs(oof_dir, exist_ok=True)
    
    model_prefix = "lgb" if model_type == "lightgbm" else "cb"
    np.save(f"{oof_dir}/{model_prefix}_oof.npy", oof)
    np.save(f"{oof_dir}/{model_prefix}_test.npy", preds)

    # Save submission
    submission = pd.DataFrame({
        "Id": test.index,
        Config.TARGET: preds
    })
    
    sub_name = "submission.csv" if model_type == "lightgbm" else "catboost_submission.csv"
    submission.to_csv(f"data/submissions/{sub_name}", index=False)

    wandb.finish()

    print(f"✅ {model_type.upper()} CV Score: {final_score:.5f}")
    return final_score


if __name__ == "__main__":
    import sys
    
    # Command line interface
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type in ["lightgbm", "lgb"]:
            run_training("lightgbm")
        elif model_type in ["catboost", "cb"]:
            run_training("catboost")
        elif model_type == "both":
            run_training("lightgbm")
            run_training("catboost")
        else:
            print("Usage: python train.py [lightgbm|catboost|both]")
    else:
        # Default: train LightGBM
        run_training("lightgbm")