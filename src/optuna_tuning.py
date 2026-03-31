# ===============================
# NEW FILE: src/optuna_tuning.py
# ===============================
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from config import Config

def objective(trial, X, y):

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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

def run_hyperparameter_tuning(X, y, n_trials=100):
    """Run Optuna hyperparameter optimization"""
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
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
    
    best_params = run_hyperparameter_tuning(X, y, n_trials=50)
    print(f"Best parameters: {best_params}")