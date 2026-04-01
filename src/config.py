# ===============================
# config.py - Enhanced Configuration
# ===============================
class Config:
    # General Settings
    TARGET = "target"
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # LightGBM Parameters
    LGB_PARAMS = {
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }
    
    # CatBoost Parameters
    CB_PARAMS = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
        "verbose": 0
    }
    
    # XGBoost Parameters (optional)
    XGB_PARAMS = {
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0
    }
    
    # Optuna Optimization Ranges
    OPTUNA_RANGES = {
        "lightgbm": {
            "learning_rate": (0.01, 0.1),
            "num_leaves": (16, 128),
            "max_depth": (3, 10),
            "min_child_samples": (10, 100),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
        },
        "catboost": {
            "learning_rate": (0.01, 0.1),
            "depth": (3, 10),
            "l2_leaf_reg": (1, 10),
            "subsample": (0.6, 1.0),
            "colsample_bylevel": (0.6, 1.0),
        },
        "xgboost": {
            "learning_rate": (0.01, 0.1),
            "max_depth": (3, 10),
            "min_child_weight": (1, 10),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0, 1),
            "reg_lambda": (0, 1),
        }
    }
    
    # Ensemble Settings
    ENSEMBLE_WEIGHTS = {
        "lightgbm": 0.33,
        "catboost": 0.33,
        "xgboost": 0.34
    }
    
    # Optuna Settings
    OPTUNA_SETTINGS = {
        "n_trials": 100,
        "cv_folds": 3,  # Use 3 folds for faster optimization
        "timeout": None,  # No timeout by default
        "direction": "maximize",
        "n_jobs": 1,  # Parallel trials (use 1 to avoid conflicts)
        "storage": None,  # Use in-memory storage by default
    }
    
    # Cross-Validation Settings
    CV_SETTINGS = {
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "stratify": True  # For classification
    }
    
    # Feature Engineering Settings
    FEATURE_SETTINGS = {
        "numeric_impute_strategy": "median",
        "categorical_impute_strategy": "most_frequent",
        "handle_unknown": "ignore",
        "drop_first": False,  # For one-hot encoding
        "sparse_features": True  # Use sparse matrices if possible
    }
    
    # File Paths
    PATHS = {
        "raw_data": "data/raw/",
        "processed_data": "data/processed/",
        "submissions": "data/submissions/",
        "oof": "data/processed/oof/",
        "models": "models/",
        "logs": "logs/"
    }
    
    # Experiment Tracking
    WANDB_SETTINGS = {
        "project": "kaggle-competition",
        "entity": None,  # Set your W&B entity/username
        "log_model": True,
        "log_artifacts": True
    }