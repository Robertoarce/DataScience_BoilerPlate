# ===============================
# NEW FILE: src/optuna_tuning.py
# ===============================
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from config import Config

def objective_lightgbm(trial, X, y):
    """LightGBM objective function"""
    ranges = Config.OPTUNA_RANGES["lightgbm"]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *ranges["learning_rate"]),
        "num_leaves": trial.suggest_int("num_leaves", *ranges["num_leaves"]),
        "max_depth": trial.suggest_int("max_depth", *ranges["max_depth"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *ranges["min_child_samples"]),
        "subsample": trial.suggest_float("subsample", *ranges["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *ranges["colsample_bytree"]),
        "n_estimators": 2000,
        "random_state": Config.RANDOM_STATE,
        "n_jobs": -1
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[tr_idx], y.iloc[tr_idx],
            eval_set=[(X[val_idx], y.iloc[val_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    return roc_auc_score(y, oof)

def objective_catboost(trial, X, y):
    """CatBoost objective function"""
    ranges = Config.OPTUNA_RANGES["catboost"]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *ranges["learning_rate"]),
        "depth": trial.suggest_int("depth", *ranges["depth"]),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *ranges["l2_leaf_reg"]),
        "subsample": trial.suggest_float("subsample", *ranges["subsample"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", *ranges["colsample_bylevel"]),
        "iterations": 2000,
        "random_state": Config.RANDOM_STATE,
        "verbose": False
    }

    skf = StratifiedKFold(n_splits=Config.OPTUNA_SETTINGS["cv_folds"], shuffle=True, random_state=Config.RANDOM_STATE)
    oof = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(
            X[tr_idx], y.iloc[tr_idx],
            eval_set=[(X[val_idx], y.iloc[val_idx])],
            early_stopping_rounds=100,
            verbose=False
        )
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    return roc_auc_score(y, oof)

def objective_xgboost(trial, X, y):
    """XGBoost objective function"""
    ranges = Config.OPTUNA_RANGES["xgboost"]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *ranges["learning_rate"]),
        "max_depth": trial.suggest_int("max_depth", *ranges["max_depth"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *ranges["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *ranges["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *ranges["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *ranges["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *ranges["reg_lambda"]),
        "n_estimators": 2000,
        "random_state": Config.RANDOM_STATE,
        "n_jobs": -1
    }

    skf = StratifiedKFold(n_splits=Config.OPTUNA_SETTINGS["cv_folds"], shuffle=True, random_state=Config.RANDOM_STATE)
    oof = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X[tr_idx], y.iloc[tr_idx],
            eval_set=[(X[val_idx], y.iloc[val_idx])],
            eval_metric="auc",
            early_stopping_rounds=100,
            verbose=False
        )
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    return roc_auc_score(y, oof)

def run_hyperparameter_tuning(X, y, model_type="lightgbm", n_trials=None):
    """Run Optuna hyperparameter optimization"""
    
    if n_trials is None:
        n_trials = Config.OPTUNA_SETTINGS["n_trials"]
    
    study = optuna.create_study(
        direction=Config.OPTUNA_SETTINGS["direction"],
        storage=Config.OPTUNA_SETTINGS["storage"]
    )
    
    # Select objective function based on model type
    if model_type == "lightgbm":
        objective_func = lambda trial: objective_lightgbm(trial, X, y)
    elif model_type == "catboost":
        objective_func = lambda trial: objective_catboost(trial, X, y)
    elif model_type == "xgboost":
        objective_func = lambda trial: objective_xgboost(trial, X, y)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    study.optimize(objective_func, n_trials=n_trials)
    
    print("Best trial:")
    print(f"Value: {study.best_trial.value}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from features import build_features
    
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    
    X, y, X_test = build_features(train, test)
    
    # Example for all models
    print("Optimizing LightGBM...")
    lgb_params = run_hyperparameter_tuning(X, y, "lightgbm", n_trials=20)
    print(f"\nLightGBM best parameters: {lgb_params}")
    
    print("\nOptimizing CatBoost...")
    cb_params = run_hyperparameter_tuning(X, y, "catboost", n_trials=20)
    print(f"CatBoost best parameters: {cb_params}")
    
    print("\nOptimizing XGBoost...")
    xgb_params = run_hyperparameter_tuning(X, y, "xgboost", n_trials=20)
    print(f"XGBoost best parameters: {xgb_params}")