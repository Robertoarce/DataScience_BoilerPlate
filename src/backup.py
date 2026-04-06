import os

import numpy as np
import optuna
import pandas as pd
import wandb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# File Paths
PATHS = {
	"raw_data": "data/raw/",
	"processed_data": "data/processed/",
	"submissions": "data/submissions/",
	"oof": "data/processed/oof/",
	"models": "models/",
	"logs": "logs/",
}


# Models Parameters
RANDOM_STATE = 42
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
	"verbose": -1,
}
CB_PARAMS = {
	"iterations": 2000,
	"learning_rate": 0.05,
	"depth": 6,
	"l2_leaf_reg": 3,
	"bootstrap_type": "Bernoulli",
	"subsample": 0.8,
	"random_state": RANDOM_STATE,
	"verbose": 0,
}
XGB_PARAMS = {
	"learning_rate": 0.05,
	"n_estimators": 2000,
	"max_depth": 6,
	"min_child_weight": 1,
	"subsample": 0.8,
	"colsample_bytree": 0.8,
	"random_state": RANDOM_STATE,
	"n_jobs": -1,
	"verbosity": 0,
}
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
	},
}
ENSEMBLE_WEIGHTS = {
	"lightgbm": 0.33,
	"catboost": 0.33,
	"xgboost": 0.34,
}
OPTUNA_SETTINGS = {
	"n_trials": 100,
	"cv_folds": 3,
	"timeout": None,
	"direction": "maximize",
	"n_jobs": 1,
	"storage": None,
}


# =========================
# CONFIG
# =========================

TARGET = "Survived"
N_SPLITS = 5
RANDOM_STATE = 42

# Experiment Tracking
WANDB_SETTINGS = {
	"base_url": "https://api.inference.wandb.ai/v1",
	"api_key": os.environ.get("WANDB_API_KEY", ""),
	"entity": "roberto_arce_",
	"project": "Titanic",
	"log_model": True,
	"log_artifacts": True,
}


# =========================
# DATA
# =========================
df = pd.read_csv(r"C:\00 ALL\05 Kaggle\02 Titanic\train.csv")


# =========================
# FEATURE ENGINEERING
# =========================


class FeatureCreator(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		# Every calculation that is column wise (to avoid data leakage)
		X = X.copy()

		# Extract titles and compute median ages for each title
		get_titles = X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
		age_frame = pd.DataFrame({
			"title": get_titles,
			"Age": X["Age"],
		})
		self.age_by_title_medians = age_frame.groupby("title")["Age"].median()

		# age median
		self.age_global_median = X["Age"].median()

		return self

	def transform(self, X):
		# Every calculation that are row wise
		X = X.copy()

		X["title"] = X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
		X["cabin_name"] = X.Cabin.str[0]

		# first by title median, then by global median
		X["Age"] = X["Age"].fillna(X["title"].map(self.age_by_title_medians))
		X["Age"] = X["Age"].fillna(self.age_global_median)

		X = X.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
		return X


# =========================
# Preprocessing Pipeline
# =========================

num_pipeline = Pipeline([
	("imputer", SimpleImputer(strategy="median")),
	("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
	("imputer", SimpleImputer(strategy="most_frequent")),
	("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessing_pipeline = ColumnTransformer([
	("num", num_pipeline, make_column_selector(dtype_include=np.number)),
	("cat", cat_pipeline, make_column_selector(dtype_include=object)),
])

full_pipeline = Pipeline([
	("features", FeatureCreator()),
	("preprocessing", preprocessing_pipeline),
	("model", RandomForestClassifier(random_state=42)),
])


# =========================
# Split Data
# =========================

y = df[TARGET].values
X = df.drop(columns=[TARGET])


# 1. Split data

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cv_splitter = StratifiedKFold(
	n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

# Baseline model
full_pipeline.fit(X_train, y_train)
baseline_pred = full_pipeline.predict(X_test)

print("Baseline holdout report")
print(classification_report(y_test, baseline_pred))
print(confusion_matrix(y_test, baseline_pred))

baseline_scores = cross_val_score(
	full_pipeline,
	X_train,
	y_train,
	cv=cv_splitter,
	scoring="f1",
)
print(f"Baseline CV F1: {baseline_scores.mean():.3f} ± {baseline_scores.std():.3f}")


# RandomizedSearchCV
param_grid = {
	"model__n_estimators": [100, 200, 300],
	"model__max_depth": [None, 10, 20],
	"model__min_samples_split": [2, 5, 10],
}

search = RandomizedSearchCV(
	full_pipeline,
	param_distributions=param_grid,
	cv=cv_splitter,
	scoring="f1",
	n_iter=20,
	random_state=RANDOM_STATE,
	n_jobs=-1,
	return_train_score=True,
	refit=True,
	verbose=1,
)
search.fit(X_train, y_train)

tuned_pred = search.predict(X_test)

print("Best parameters:", search.best_params_)
print(f"Best CV F1: {search.best_score_:.3f}")
print(f"Search score on test (F1): {search.score(X_test, y_test):.3f}")
print(f"Best estimator test accuracy: {search.best_estimator_.score(X_test, y_test):.3f}")
print(f"Best estimator test F1: {f1_score(y_test, tuned_pred):.3f}")
print(classification_report(y_test, tuned_pred))
print(confusion_matrix(y_test, tuned_pred))


# =========================
# FEATURE IMPORTANCE
# =========================
best_pipeline = search.best_estimator_

X_features = best_pipeline.named_steps["features"].transform(X_train)
feature_names = best_pipeline.named_steps["preprocessing"].get_feature_names_out(
	X_features.columns
)
importances = best_pipeline.named_steps["model"].feature_importances_

fi = (
	pd.DataFrame(
		{
			"feature": feature_names,
			"importance": importances,
		}
	)
	.sort_values(by="importance", ascending=False)
	.reset_index(drop=True)
)

print(fi.head(20))


# =========================
# MODEL BENCHMARKING
# =========================
benchmark_models = {
	"random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
	"logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
	"knn": KNeighborsClassifier(),
	"svc": SVC(),
	"decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
}

benchmark_results = []
wandb_run = None

if WANDB_SETTINGS.get("api_key"):
	try:
		os.environ["WANDB_API_KEY"] = WANDB_SETTINGS["api_key"]
		wandb.login(key=WANDB_SETTINGS["api_key"], relogin=True)
		wandb_run = wandb.init(
			project=WANDB_SETTINGS["project"],
			entity=WANDB_SETTINGS["entity"],
			config={
				"benchmark_models": list(benchmark_models.keys()),
				"cv_folds": N_SPLITS,
				"scoring": "f1",
			},
			job_type="benchmark",
			reinit=True,
		)
	except Exception as exc:
		print(f"W&B logging skipped: {exc}")

for model_name, model in benchmark_models.items():
	pipeline = Pipeline([
		("features", FeatureCreator()),
		("preprocessing", preprocessing_pipeline),
		("model", model),
	])

	cv_scores = cross_val_score(
		pipeline,
		X_train,
		y_train,
		cv=cv_splitter,
		scoring="f1",
		n_jobs=-1,
	)

	pipeline.fit(X_train, y_train)
	holdout_pred = pipeline.predict(X_test)

	result = {
		"model": model_name,
		"cv_f1_mean": cv_scores.mean(),
		"cv_f1_std": cv_scores.std(),
		"holdout_f1": f1_score(y_test, holdout_pred),
		"holdout_accuracy": accuracy_score(y_test, holdout_pred),
	}
	benchmark_results.append(result)

	if wandb_run is not None:
		wandb.log({f"benchmark/{model_name}/{key}": value for key, value in result.items() if key != "model"})

benchmark_df = pd.DataFrame(benchmark_results).sort_values(
	by=["holdout_f1", "cv_f1_mean"], ascending=False
).reset_index(drop=True)

print(benchmark_df)

if wandb_run is not None:
	wandb.log({"benchmark_table": wandb.Table(dataframe=benchmark_df)})
	wandb.finish()
