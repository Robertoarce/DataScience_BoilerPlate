import os
from pathlib import Path

import numpy as np 
import pandas as pd
from dotenv import load_dotenv

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from wandb_utils import (
	build_feature_importance_frame,
	build_permutation_importance_frame,
	build_shap_artifacts,
	build_tree_graphviz_artifacts,
	extract_prediction_scores,
	finish_wandb_run,
	initialize_wandb_run,
	log_baseline_results,
	log_benchmark_model_results,
	log_benchmark_summary_results,
	log_feature_importance_results,
	log_model_artifact,
	log_model_search_results,
	log_permutation_importance_results,
	log_shap_results,
	log_tree_graphviz_results,
	log_search_comparison_results,
	log_tuning_results,
)
from model_io import save_model


CURRENT_DIR = Path(__file__).resolve().parent
load_dotenv(CURRENT_DIR / ".env")
load_dotenv(CURRENT_DIR.parent / ".env")


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


MODEL_SEARCH_CONFIGS = {
	"random_forest": {
		"search_type": "random",
		"params": {
			"model__n_estimators": [100, 200, 300, 500],
			"model__max_depth": [None, 5, 10, 20],
			"model__min_samples_split": [2, 5, 10],
			"model__min_samples_leaf": [1, 2, 4],
			"model__max_features": ["sqrt", "log2", None],
		},
		"n_iter": 15,
	},
	"extra_trees": {
		"search_type": "random",
		"params": {
			"model__n_estimators": [100, 200, 300, 500],
			"model__max_depth": [None, 5, 10, 20],
			"model__min_samples_split": [2, 5, 10],
			"model__min_samples_leaf": [1, 2, 4],
			"model__max_features": ["sqrt", "log2", None],
		},
		"n_iter": 15,
	},
	"xgboost": {
		"search_type": "random",
		"params": {
			"model__n_estimators": [100, 200, 300, 500],
			"model__learning_rate": [0.01, 0.05, 0.1],
			"model__max_depth": [3, 4, 6, 8],
			"model__min_child_weight": [1, 3, 5],
			"model__subsample": [0.7, 0.8, 1.0],
			"model__colsample_bytree": [0.7, 0.8, 1.0],
		},
		"n_iter": 15,
	},
	"logistic_regression": {
		"search_type": "grid",
		"params": {
			"model__C": [0.01, 0.1, 1.0, 10.0],
			"model__class_weight": [None, "balanced"],
		},
	},
	"ridge_classifier": {
		"search_type": "grid",
		"params": {
			"model__alpha": [0.1, 1.0, 10.0],
			"model__class_weight": [None, "balanced"],
		},
	},
	"knn": {
		"search_type": "grid",
		"params": {
			"model__n_neighbors": [3, 5, 7, 9],
			"model__weights": ["uniform", "distance"],
			"model__p": [1, 2],
		},
	},
	"svc": {
		"search_type": "grid",
		"params": {
			"model__C": [0.5, 1.0, 5.0],
			"model__kernel": ["linear", "rbf"],
			"model__gamma": ["scale"],
			"model__class_weight": [None, "balanced"],
		},
	},
	"linear_svc": {
		"search_type": "grid",
		"params": {
			"model__C": [0.1, 1.0, 10.0],
			"model__class_weight": [None, "balanced"],
		},
	},
	"decision_tree": {
		"search_type": "grid",
		"params": {
			"model__max_depth": [None, 5, 10],
			"model__min_samples_split": [2, 5],
			"model__min_samples_leaf": [1, 2],
			"model__ccp_alpha": [0.0, 0.01],
			"model__class_weight": [None, "balanced"],
		},
	},
}


# =========================
# DATA
# =========================
df = pd.read_csv(r"C:\00 ALL\05 Kaggle\02 Titanic\train.csv")


# =========================
# FEATURE ENGINEERING
# =========================


class FeatureCreator(BaseEstimator, TransformerMixin):
	def fit(self, X, _y=None):
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
], memory=None)

benchmark_models = {
	"random_forest": RandomForestClassifier(
		random_state=RANDOM_STATE,
		n_estimators=100,
		min_samples_leaf=1,
		max_features="sqrt",
	),
	"extra_trees": ExtraTreesClassifier(
		random_state=RANDOM_STATE,
		n_estimators=200,
		min_samples_leaf=1,
		max_features="sqrt",
	),
	"xgboost": XGBClassifier(
		objective="binary:logistic",
		eval_metric="logloss",
		random_state=RANDOM_STATE,
		n_estimators=200,
		learning_rate=0.05,
		max_depth=4,
		min_child_weight=1,
		subsample=0.8,
		colsample_bytree=0.8,
		n_jobs=-1,
	),
	"logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
	"ridge_classifier": RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE),
	"knn": KNeighborsClassifier(n_neighbors=5),
	"svc": SVC(C=1.0, kernel="rbf", gamma="scale"),
	"linear_svc": LinearSVC(C=1.0, random_state=RANDOM_STATE, dual="auto"),
	"decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=0.0),
}



def build_model_pipeline(model):
	return Pipeline([
		("features", FeatureCreator()),
		("preprocessing", preprocessing_pipeline),
		("model", clone(model)),
	], memory=None)


def build_transformed_feature_frame(fitted_pipeline, features):
	engineered_features = fitted_pipeline.named_steps["features"].transform(features)
	feature_names = fitted_pipeline.named_steps["preprocessing"].get_feature_names_out(
		engineered_features.columns
	)
	transformed_features = fitted_pipeline.named_steps["preprocessing"].transform(engineered_features)

	if hasattr(transformed_features, "toarray"):
		transformed_features = transformed_features.toarray()

	transformed_feature_frame = pd.DataFrame(
		transformed_features,
		columns=feature_names,
		index=engineered_features.index,
	)
	return engineered_features, transformed_feature_frame, feature_names


def run_model_search(model_name, model, search_config, X_train, y_train, X_test, y_test, search_splitter, include_shap=False):
	pipeline = build_model_pipeline(model)

	if search_config["search_type"] == "random":
		search = RandomizedSearchCV(
			pipeline,
			param_distributions=search_config["params"],
			n_iter=search_config.get("n_iter", 10),
			cv=search_splitter,
			scoring="f1",
			random_state=RANDOM_STATE,
			n_jobs=-1,
			return_train_score=True,
			refit=True,
			verbose=1,
		)
	else:
		search = GridSearchCV(
			pipeline,
			param_grid=search_config["params"],
			cv=search_splitter,
			scoring="f1",
			n_jobs=-1,
			return_train_score=True,
			refit=True,
			verbose=1,
		)

	search.fit(X_train, y_train)
	y_pred = search.predict(X_test)
	y_scores = extract_prediction_scores(search.best_estimator_, X_test)

	_, transformed_train_feature_frame, feature_names = build_transformed_feature_frame(search.best_estimator_, X_train)
	_, transformed_test_feature_frame, _ = build_transformed_feature_frame(search.best_estimator_, X_test)
	feature_importance_df = build_feature_importance_frame(
		model_name,
		search.best_estimator_.named_steps["model"],
		feature_names,
	)
	permutation_importance_df = build_permutation_importance_frame(
		model_name,
		search.best_estimator_,
		X_test,
		y_test,
		random_state=RANDOM_STATE,
		n_repeats=5,
	)
	transformed_permutation_importance_df = build_permutation_importance_frame(
		model_name,
		search.best_estimator_.named_steps["model"],
		transformed_test_feature_frame.to_numpy(),
		y_test,
		random_state=RANDOM_STATE,
		n_repeats=5,
		feature_names=transformed_test_feature_frame.columns,
	)
	shap_artifacts = None
	if include_shap:
		shap_artifacts = build_shap_artifacts(
			model_name,
			search.best_estimator_.named_steps["model"],
			transformed_train_feature_frame,
			transformed_test_feature_frame,
			random_state=RANDOM_STATE,
		)
	tree_graphviz_artifacts = build_tree_graphviz_artifacts(
		model_name,
		search.best_estimator_.named_steps["model"],
		feature_names,
	)

	result = {
		"model": model_name,
		"search_type": search_config["search_type"],
		"best_cv_f1": search.best_score_,
		"holdout_f1": f1_score(y_test, y_pred),
		"holdout_accuracy": accuracy_score(y_test, y_pred),
	}
	if y_scores is not None:
		result["holdout_roc_auc"] = roc_auc_score(y_test, y_scores)

	return {
		"search": search,
		"result": result,
		"y_pred": y_pred,
		"y_scores": y_scores,
		"feature_importance": feature_importance_df,
		"permutation_importance": permutation_importance_df,
		"transformed_permutation_importance": transformed_permutation_importance_df,
		"shap_artifacts": shap_artifacts,
		"tree_graphviz_artifacts": tree_graphviz_artifacts,
	}


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

search_cv_splitter = StratifiedKFold(
	n_splits=OPTUNA_SETTINGS["cv_folds"], shuffle=True, random_state=RANDOM_STATE
)

wandb_run = initialize_wandb_run(
	settings=WANDB_SETTINGS,
	target=TARGET,
	random_state=RANDOM_STATE,
	n_splits=N_SPLITS,
	paths=PATHS,
	benchmark_models=benchmark_models,
	baseline_model=full_pipeline.named_steps["model"],
)

# Baseline model
full_pipeline.fit(X_train, y_train)
baseline_pred = full_pipeline.predict(X_test)
baseline_scores_pred = extract_prediction_scores(full_pipeline, X_test)

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

if wandb_run is not None:
	log_baseline_results(wandb_run, y_test, baseline_pred, baseline_scores, baseline_scores_pred)


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
tuned_scores_pred = extract_prediction_scores(search.best_estimator_, X_test)

print("\n\n--------- RandomizedSearchCV Results ---------")
print("Best parameters:", search.best_params_)
print(f"Best CV F1: {search.best_score_:.3f}")
print(f"Search score on test (F1): {search.score(X_test, y_test):.3f}")
print(f"Best estimator test accuracy: {search.best_estimator_.score(X_test, y_test):.3f}")
print(f"Best estimator test F1: {f1_score(y_test, tuned_pred):.3f}")
print(classification_report(y_test, tuned_pred))
print(confusion_matrix(y_test, tuned_pred))

if wandb_run is not None:
	log_tuning_results(wandb_run, search, param_grid, X_test, y_test, tuned_pred, tuned_scores_pred)

# =========================
# FEATURE IMPORTANCE
# =========================
best_pipeline = search.best_estimator_

X_features, transformed_X_train_features, feature_names = build_transformed_feature_frame(best_pipeline, X_train)
_, transformed_X_test_features, _ = build_transformed_feature_frame(best_pipeline, X_test)
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

print("\n\n--------- Feature Importance Results ---------")
print(fi.head(20))

permutation_fi = build_permutation_importance_frame(
	"random_forest_tuned",
	search.best_estimator_,
	X_test,
	y_test,
	random_state=RANDOM_STATE,
	n_repeats=5,
)

transformed_permutation_fi = build_permutation_importance_frame(
	"random_forest_tuned_transformed",
	search.best_estimator_.named_steps["model"],
	transformed_X_test_features.to_numpy(),
	y_test,
	random_state=RANDOM_STATE,
	n_repeats=5,
	feature_names=transformed_X_test_features.columns,
)

tuned_shap_artifacts = build_shap_artifacts(
	"random_forest_tuned",
	search.best_estimator_.named_steps["model"],
	transformed_X_train_features,
	transformed_X_test_features,
	random_state=RANDOM_STATE,
)

tuned_tree_graphviz_artifacts = build_tree_graphviz_artifacts(
	"random_forest_tuned",
	search.best_estimator_.named_steps["model"],
	feature_names,
)

print("\n\n--------- Raw Feature Permutation Importance Results ---------")
print(permutation_fi.head(10))

print("\n\n--------- Transformed Feature Permutation Importance Results ---------")
print(transformed_permutation_fi.head(10).to_string(index=False))

if tuned_shap_artifacts is not None:
	print("\n\n--------- SHAP Results ---------")
	print(tuned_shap_artifacts["summary_df"].head(10).to_string(index=False))

if wandb_run is not None:
	log_feature_importance_results(
		wandb_run,
		"tuning/feature_importance",
		fi,
		"Tuned Model Feature Importance (Top 20)",
	)
	log_permutation_importance_results(
		wandb_run,
		"tuning/permutation_importance_raw",
		permutation_fi,
		"Tuned Model Raw Feature Permutation Importance (Top 20)",
	)
	log_permutation_importance_results(
		wandb_run,
		"tuning/permutation_importance_transformed",
		transformed_permutation_fi,
		"Tuned Model Transformed Feature Permutation Importance (Top 20)",
	)
	log_shap_results(
		wandb_run,
		"tuning/shap",
		tuned_shap_artifacts,
		"Tuned Model SHAP Mean Absolute Impact (Top 20)",
	)
	log_tree_graphviz_results(
		wandb_run,
		"tuning/tree",
		tuned_tree_graphviz_artifacts,
	)


# =========================
# MODELS BENCHMARKING
# =========================
benchmark_results = []

for model_name, model in benchmark_models.items():
	pipeline = Pipeline([
		("features", FeatureCreator()),
		("preprocessing", preprocessing_pipeline),
		("model", model),
	], memory=None)

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
	holdout_scores = extract_prediction_scores(pipeline, X_test)
	feature_frame = pipeline.named_steps["features"].transform(X_train)
	benchmark_feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out(
		feature_frame.columns
	)
	benchmark_feature_importance = build_feature_importance_frame(
		model_name,
		pipeline.named_steps["model"],
		benchmark_feature_names,
	)
	benchmark_permutation_importance = build_permutation_importance_frame(
		model_name,
		pipeline,
		X_test,
		y_test,
		random_state=RANDOM_STATE,
		n_repeats=5,
	)
	benchmark_tree_graphviz_artifacts = build_tree_graphviz_artifacts(
		model_name,
		pipeline.named_steps["model"],
		benchmark_feature_names,
	)

	result = {
		"model": model_name,
		"cv_f1_mean": cv_scores.mean(),
		"cv_f1_std": cv_scores.std(),
		"holdout_f1": f1_score(y_test, holdout_pred),
		"holdout_accuracy": accuracy_score(y_test, holdout_pred),
	}
	if holdout_scores is not None:
		result["holdout_roc_auc"] = roc_auc_score(y_test, holdout_scores)
	benchmark_results.append(result)

	if wandb_run is not None:
		log_benchmark_model_results(
			wandb_run,
			model_name,
			model,
			result,
			cv_scores,
			y_test,
			holdout_pred,
			holdout_scores,
			benchmark_feature_importance,
			benchmark_permutation_importance,
			benchmark_tree_graphviz_artifacts,
		)

benchmark_df = pd.DataFrame(benchmark_results).sort_values(
	by=["holdout_f1", "cv_f1_mean"], ascending=False
).reset_index(drop=True)

print("\n\n--------- Benchmark Results ---------")
print(benchmark_df)

best_benchmark_model_name = benchmark_df.iloc[0]["model"]
print("\n\n--------- Best Benchmark Model ---------")
print(f"Selected model for tuning: {best_benchmark_model_name}")

best_model_search = run_model_search(
	best_benchmark_model_name,
	benchmark_models[best_benchmark_model_name],
	MODEL_SEARCH_CONFIGS[best_benchmark_model_name],
	X_train,
	y_train,
	X_test,
	y_test,
	search_cv_splitter,
	include_shap=True,
)

print("\n\n--------- Best Benchmark Model Tuned Results ---------")
print("Best parameters:", best_model_search["search"].best_params_)
print(f"Best CV F1: {best_model_search['result']['best_cv_f1']:.3f}")
print(f"Holdout accuracy: {best_model_search['result']['holdout_accuracy']:.3f}")
print(f"Holdout F1: {best_model_search['result']['holdout_f1']:.3f}")
if "holdout_roc_auc" in best_model_search["result"]:
	print(f"Holdout ROC AUC: {best_model_search['result']['holdout_roc_auc']:.3f}")

if best_model_search["feature_importance"] is not None:
	print(f"\nTop model-specific importance features for {best_benchmark_model_name}:")
	print(best_model_search["feature_importance"].head(10))

print(f"\nTop raw permutation importance features for {best_benchmark_model_name}:")
print(best_model_search["permutation_importance"].head(10))

print(f"\nTop transformed permutation importance features for {best_benchmark_model_name}:")
print(best_model_search["transformed_permutation_importance"].head(10).to_string(index=False))

if best_model_search["shap_artifacts"] is not None:
	print(f"\nTop SHAP features for {best_benchmark_model_name}:")
	print(best_model_search["shap_artifacts"]["summary_df"].head(10).to_string(index=False))

family_search_results = []
family_search_runs = {}

for model_name in [best_benchmark_model_name] + [
	name for name in MODEL_SEARCH_CONFIGS if name != best_benchmark_model_name
]:
	if model_name == best_benchmark_model_name:
		model_search = best_model_search
	else:
		model_search = run_model_search(
			model_name,
			benchmark_models[model_name],
			MODEL_SEARCH_CONFIGS[model_name],
			X_train,
			y_train,
			X_test,
			y_test,
			search_cv_splitter,
		)

	family_search_runs[model_name] = model_search
	family_search_results.append(model_search["result"])

	if wandb_run is not None:
		log_model_search_results(
			wandb_run,
			f"family_search/{model_name}",
			model_search["search"],
			model_search["result"],
			y_test,
			model_search["y_pred"],
			model_search["y_scores"],
			model_search["feature_importance"],
			model_search["permutation_importance"],
			model_search["transformed_permutation_importance"],
			model_search["shap_artifacts"],
			model_search["tree_graphviz_artifacts"],
		)

family_search_df = pd.DataFrame(family_search_results).sort_values(
	by=["holdout_f1", "best_cv_f1"], ascending=False
).reset_index(drop=True)

print("\n\n--------- Family Search Comparison Results ---------")
print(family_search_df)

saved_family_model_paths = {}
for model_name, model_search in family_search_runs.items():
	saved_family_model_paths[model_name] = save_model(
		model_search["search"].best_estimator_,
		model_type=f"family_search_{model_name}",
	)

saved_family_models_df = pd.DataFrame(
	[
		{
			"model": model_name,
			"model_path": model_path,
		}
		for model_name, model_path in saved_family_model_paths.items()
	]
).sort_values(by="model").reset_index(drop=True)

print("\n\n--------- Saved Family Search Models ---------")
print(saved_family_models_df.to_string(index=False))

best_family_model_name = family_search_df.iloc[0]["model"]
best_family_model_search = family_search_runs[best_family_model_name]
saved_model_path = save_model(best_family_model_search["search"].best_estimator_, model_type="best_family_search")

print("\n\n--------- Saved Final Model ---------")
print(f"Selected final model family: {best_family_model_name}")
print(f"Saved model path: {saved_model_path}")

if wandb_run is not None:
	log_benchmark_summary_results(wandb_run, benchmark_df)
	log_search_comparison_results(wandb_run, "family_search", family_search_df, "Family Search")
	for model_name, model_path in saved_family_model_paths.items():
		model_search = family_search_runs[model_name]
		log_model_artifact(
			wandb_run,
			model_path,
			f"titanic-family-search-{model_name}-model",
			metadata={
				"model_family": model_name,
				"result": model_search["result"],
				"best_params": model_search["search"].best_params_,
			},
			aliases=["latest", "family-search", model_name],
		)
	log_model_artifact(
		wandb_run,
		saved_model_path,
		"titanic-best-family-search-model",
		metadata={
			"selected_model_family": best_family_model_name,
			"result": best_family_model_search["result"],
			"best_params": best_family_model_search["search"].best_params_,
		},
		aliases=["latest", "best", best_family_model_name],
	)
	finish_wandb_run(wandb_run)
