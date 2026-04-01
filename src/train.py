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
import xgboost as xgb
import os

from config import Config
from features import build_features
from utils import plot_feature_importance, save_model, load_model
from optuna_tuning import run_hyperparameter_tuning



def get_model(model_type="lightgbm", optimized_params=None):
    """Get model based on type with optional optimized parameters"""
    
    # Use optimized params if available, otherwise use defaults
    if optimized_params is not None:
        if model_type == "lightgbm":
            # Merge optimized params with base params
            params = {**Config.LGB_PARAMS, **optimized_params}
            return lgb.LGBMClassifier(**params)
        elif model_type == "catboost":
            params = {**Config.CB_PARAMS, **optimized_params}
            return CatBoostClassifier(**params)
        elif model_type == "xgboost":
            params = {**Config.XGB_PARAMS, **optimized_params}
            return xgb.XGBClassifier(**params)
    else:
        # Use default parameters
        if model_type == "lightgbm":
            return lgb.LGBMClassifier(**Config.LGB_PARAMS)
        elif model_type == "catboost":
            return CatBoostClassifier(**Config.CB_PARAMS)
        elif model_type == "xgboost":
            return xgb.XGBClassifier(**Config.XGB_PARAMS)
    
    raise ValueError(f"Unknown model type: {model_type}")

def run_training(model_type="lightgbm", use_optuna=True, n_trials=None, feature_version="v1", use_feature_cache=True):
    """
    Unified training pipeline with feature versioning
    
    Args:
        model_type: "lightgbm", "catboost", or "xgboost"
        use_optuna: Whether to use hyperparameter optimization
        n_trials: Number of Optuna trials
        feature_version: Feature version ("v1", "v2", "v3")
        use_feature_cache: Whether to use cached features
    """
    
    # W&B project name based on model and feature version
    project_name = f"kaggle-{model_type}-{feature_version}"
    if model_type == "lightgbm":
        model_config = Config.LGB_PARAMS
    elif model_type == "catboost":
        model_config = Config.CB_PARAMS
    else:  # xgboost
        model_config = Config.XGB_PARAMS
    
    # Add feature version to config
    model_config = {**model_config, "feature_version": feature_version}
    wandb.init(project=project_name, config=model_config)

    print(f"🚀 Training {model_type.upper()} model with features {feature_version}...")
    
    # Load data
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    
    # Build features with versioning
    try:
        X, y, X_test, feature_names = build_features(train, test, version=feature_version, use_cache=use_feature_cache)
        print(f"   Features shape: {X.shape}, Test shape: {X_test.shape}")
        if feature_names:
            print(f"   Feature names available: {len(feature_names)} features")
    except ValueError as e:
        if "Unknown feature version" in str(e):
            print(f"❌ {e}")
            print("   Available versions: v1, v2, v3, v4")
            return None
        raise

    # Optuna hyperparameter optimization
    optimized_params = None
    if use_optuna:
        print(f"🔍 Running Optuna hyperparameter optimization for {model_type.upper()}...")
        
        trials = n_trials or Config.OPTUNA_SETTINGS["n_trials"]
        print(f"   Running {trials} trials with {Config.OPTUNA_SETTINGS['cv_folds']}-fold CV...")
        
        try:
            optimized_params = run_hyperparameter_tuning(X, y, model_type, trials)
            print(f"✓ Optuna optimization completed!")
            print("📈 Best parameters:")
            for key, value in optimized_params.items():
                print(f"   {key}: {value}")
            
            # Log optimized params to W&B
            wandb.log({"optimized_params": optimized_params})
            
        except Exception as e:
            print(f"⚠️ Optuna optimization failed: {e}")
            print("   Continuing with default parameters...")
            optimized_params = None
    else:
        print(f"⚡ Skipping hyperparameter optimization, using default parameters...")

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

        model = get_model(model_type, optimized_params)

        # Model-specific training
        if model_type == "lightgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        elif model_type == "catboost":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
        else:  # xgboost
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
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
        elif model_type == "xgboost":
            feature_importance.append(model.feature_importances_)

        score = roc_auc_score(y_val, oof[val_idx])
        wandb.log({f"fold_{fold}": score})
        print(f"Fold {fold}: {score:.5f}")

    final_score = roc_auc_score(y, oof)
    wandb.log({"oof_score": final_score})

    # Feature importance analysis
    print("📊 Analyzing feature importance...")
    avg_importance = np.mean(feature_importance, axis=0)
    
    # Plot and save feature importance
    feat_imp_df = plot_feature_importance(models[-1], feature_names, model_type)
    
    # Save feature importance to CSV
    os.makedirs("plots", exist_ok=True)
    feat_imp_df.to_csv(f"plots/{model_type}_{feature_version}_feature_importance.csv", index=False)
    
    # Log feature importance to W&B
    wandb.log({"feature_importance": wandb.Table(dataframe=feat_imp_df.head(20))})

    # Save final model (last fold)
    save_model(models[-1], f"{model_type}_{feature_version}")

    # Save OOF predictions with feature version
    oof_dir = "data/processed/oof/"
    os.makedirs(oof_dir, exist_ok=True)
    
    if model_type == "lightgbm":
        model_prefix = "lgb"
    elif model_type == "catboost":
        model_prefix = "cb"
    else:  # xgboost
        model_prefix = "xgb"
    
    np.save(f"{oof_dir}/{model_prefix}_{feature_version}_oof.npy", oof)
    np.save(f"{oof_dir}/{model_prefix}_{feature_version}_test.npy", preds)

    # Save submission with feature version
    submission = pd.DataFrame({
        "Id": test.index,
        Config.TARGET: preds
    })
    
    os.makedirs("data/submissions", exist_ok=True)
    if model_type == "lightgbm":
        sub_name = f"submission_{feature_version}.csv"
    elif model_type == "catboost":
        sub_name = f"catboost_{feature_version}_submission.csv"
    else:  # xgboost
        sub_name = f"xgboost_{feature_version}_submission.csv"
    
    submission.to_csv(f"data/submissions/{sub_name}", index=False)

    wandb.finish()

    print(f"✅ {model_type.upper()} CV Score: {final_score:.5f}")
    return final_score


if __name__ == "__main__":
    import sys
    import argparse
    
    # Command line interface with argparse
    parser = argparse.ArgumentParser(description="Train ML models with optional Optuna hyperparameter optimization")
    parser.add_argument("model", nargs='?', default="lightgbm", 
                       choices=["lightgbm", "lgb", "catboost", "cb", "xgboost", "xgb", "both", "all"],
                       help="Model type to train")
    parser.add_argument("--no-optuna", action="store_true", 
                       help="Disable Optuna hyperparameter optimization (use default params)")
    parser.add_argument("--trials", type=int, default=None,
                       help=f"Number of Optuna trials (default: {Config.OPTUNA_SETTINGS['n_trials']})")
    parser.add_argument("--features", "--version", default="v1", 
                       choices=["v1", "v2", "v3", "v4"],
                       help="Feature version to use (default: v1)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable feature caching (rebuild features each time)")
    
    # For backward compatibility, also support old-style args
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
        # Old style: python train.py lightgbm
        model_type = sys.argv[1].lower()
        use_optuna = True  # Default to using Optuna
        n_trials = None
        feature_version = "v1"  # Default to v1
        use_feature_cache = True
        
        # Check for additional flags
        if "--no-optuna" in sys.argv:
            use_optuna = False
        if "--trials" in sys.argv:
            trials_index = sys.argv.index("--trials") + 1
            if trials_index < len(sys.argv):
                try:
                    n_trials = int(sys.argv[trials_index])
                except ValueError:
                    print("Invalid trials value, using default...")
                    n_trials = None
        if "--features" in sys.argv:
            features_index = sys.argv.index("--features") + 1
            if features_index < len(sys.argv):
                feature_version = sys.argv[features_index]
        if "--no-cache" in sys.argv:
            use_feature_cache = False
    else:
        # New style with argparse
        args = parser.parse_args()
        model_type = args.model
        use_optuna = not args.no_optuna
        n_trials = args.trials
        feature_version = args.features
        use_feature_cache = not args.no_cache
    
    # Execute training based on model type
    if model_type in ["lightgbm", "lgb"]:
        run_training("lightgbm", use_optuna=use_optuna, n_trials=n_trials, 
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
    elif model_type in ["catboost", "cb"]:
        run_training("catboost", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache) 
    elif model_type in ["xgboost", "xgb"]:
        run_training("xgboost", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
    elif model_type == "all":
        print(f"🚀 Training all models with features {feature_version}...")
        run_training("lightgbm", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
        run_training("catboost", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
        run_training("xgboost", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
    elif model_type == "both":  # backward compatibility
        print(f"🚀 Training LightGBM and CatBoost with features {feature_version}...")
        run_training("lightgbm", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
        run_training("catboost", use_optuna=use_optuna, n_trials=n_trials,
                    feature_version=feature_version, use_feature_cache=use_feature_cache)
    else:
        parser.print_help()