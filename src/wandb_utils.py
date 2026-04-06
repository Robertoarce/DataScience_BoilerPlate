import os

import numpy as np
import pandas as pd
import wandb

from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, roc_curve


CLASS_NAMES = ["Not Survived", "Survived"]


def serialize_for_wandb(value):
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	if isinstance(value, (np.integer, np.floating, np.bool_)):
		return value.item()
	if isinstance(value, dict):
		return {str(key): serialize_for_wandb(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [serialize_for_wandb(item) for item in value]
	return str(value)


def build_model_config_rows(model_name, model, context):
	return [
		{
			"context": context,
			"model": model_name,
			"parameter": parameter,
			"value": str(serialize_for_wandb(value)),
		}
		for parameter, value in sorted(model.get_params(deep=False).items())
	]


def initialize_wandb_run(
	settings,
	target,
	random_state,
	n_splits,
	paths,
	benchmark_models,
	baseline_model,
	test_size=0.2,
):
	if not settings.get("api_key"):
		print("W&B logging skipped: WANDB_API_KEY not configured")
		return None
 

	try:
		os.environ["WANDB_API_KEY"] = settings["api_key"]
		wandb.login(key=settings["api_key"], relogin=True)
		run = wandb.init(
			project=settings["project"],
			entity=settings["entity"],
			config={
				"target": target,
				"random_state": random_state,
				"n_splits": n_splits,
				"test_size": test_size,
				"paths": serialize_for_wandb(paths),
				"benchmark_models": list(benchmark_models.keys()),
			},
			job_type="training",
			reinit=True,
		)

		run.config.update(
			{
				"wandb_settings": serialize_for_wandb(settings), 
			},
			allow_val_change=True,
		)

		run.log(
			{ 
				"metadata/model_configs": wandb.Table(
					dataframe=pd.DataFrame(
						build_model_config_rows("baseline_random_forest", baseline_model, "baseline")
						+ [
							row
							for model_name, model in benchmark_models.items()
							for row in build_model_config_rows(model_name, model, "benchmark")
						]
					)
				),
			}
		)
		return run
	except Exception as exc:
		print(f"W&B logging skipped: {exc}")
		return None


def extract_prediction_scores(model, features):
	if hasattr(model, "predict_proba"):
		probabilities = model.predict_proba(features)
		if probabilities.ndim == 2 and probabilities.shape[1] > 1:
			return probabilities[:, 1]
		return probabilities.ravel()
	if hasattr(model, "decision_function"):
		scores = model.decision_function(features)
		if np.ndim(scores) > 1:
			return scores[:, 1]
		return scores
	return None


def build_feature_importance_frame(model_name, model, feature_names):
	if hasattr(model, "feature_importances_"):
		importance_values = np.asarray(model.feature_importances_)
	elif hasattr(model, "coef_"):
		coefficients = np.asarray(model.coef_)
		importance_values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
	else:
		return None

	return (
		pd.DataFrame(
			{
				"model": model_name,
				"feature": feature_names,
				"importance": importance_values,
			}
		)
		.sort_values(by="importance", ascending=False)
		.reset_index(drop=True)
	)


def log_table_with_bar_chart(run, key, dataframe, x_column, y_column, title):
	table = wandb.Table(dataframe=dataframe)
	run.log(
		{
			f"{key}/table": table,
			f"{key}/bar_chart": wandb.plot.bar(table, x_column, y_column, title=title),
		}
	)


def log_table_with_line_chart(run, key, dataframe, x_column, y_column, title):
	table = wandb.Table(dataframe=dataframe)
	run.log(
		{
			f"{key}/table": table,
			f"{key}/line_chart": wandb.plot.line(table, x_column, y_column, title=title),
		}
	)


def log_metrics_visuals(run, key, y_true, y_pred, y_scores=None):
	run.log(
		{
			f"{key}/confusion_matrix": wandb.plot.confusion_matrix(
				probs=None,
				y_true=y_true.tolist(),
				preds=y_pred.tolist(),
				class_names=CLASS_NAMES,
			),
		}
	)

	if y_scores is None:
		return

	fpr, tpr, _ = roc_curve(y_true, y_scores)
	roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
	roc_table = wandb.Table(dataframe=roc_df)
	run.log(
		{
			f"{key}/roc_curve_table": roc_table,
			f"{key}/roc_curve": wandb.plot.line(roc_table, "fpr", "tpr", title=f"{key} ROC Curve"),
			f"{key}/roc_auc": auc(fpr, tpr),
			f"{key}/roc_auc_score": roc_auc_score(y_true, y_scores),
		}
	)


def log_baseline_results(run, y_test, baseline_pred, baseline_scores, baseline_scores_pred):
	baseline_cv_df = pd.DataFrame(
		{
			"fold": np.arange(1, len(baseline_scores) + 1),
			"f1": baseline_scores,
		}
	)
	run.log(
		{
			"baseline/holdout_accuracy": accuracy_score(y_test, baseline_pred),
			"baseline/holdout_f1": f1_score(y_test, baseline_pred),
			"baseline/cv_f1_mean": baseline_scores.mean(),
			"baseline/cv_f1_std": baseline_scores.std(),
		}
	)
	log_table_with_line_chart(
		run,
		"baseline/cv_folds",
		baseline_cv_df,
		"fold",
		"f1",
		"Baseline CV F1 by Fold",
	)
	log_metrics_visuals(run, "baseline", y_test, baseline_pred, baseline_scores_pred)


def log_tuning_results(run, search, param_grid, X_test, y_test, tuned_pred, tuned_scores_pred):
	search_results_df = (
		pd.DataFrame(
			{
				"rank_test_score": search.cv_results_["rank_test_score"],
				"mean_test_score": search.cv_results_["mean_test_score"],
				"std_test_score": search.cv_results_["std_test_score"],
				"mean_train_score": search.cv_results_["mean_train_score"],
				"n_estimators": search.cv_results_["param_model__n_estimators"],
				"max_depth": search.cv_results_["param_model__max_depth"],
				"min_samples_split": search.cv_results_["param_model__min_samples_split"],
			}
		)
		.sort_values(by="rank_test_score")
		.reset_index(drop=True)
	)
	run.config.update(
		{
			"search_space": serialize_for_wandb(param_grid),
			"random_search_best_params": serialize_for_wandb(search.best_params_),
			"random_search_best_score": search.best_score_,
		},
		allow_val_change=True,
	)
	run.log(
		{
			"tuning/best_cv_f1": search.best_score_,
			"tuning/holdout_accuracy": search.best_estimator_.score(X_test, y_test),
			"tuning/holdout_f1": f1_score(y_test, tuned_pred),
			"tuning/search_results_table": wandb.Table(dataframe=search_results_df),
			"tuning/best_model_config": wandb.Table(
				dataframe=pd.DataFrame(
					build_model_config_rows(
						"random_forest_tuned",
						search.best_estimator_.named_steps["model"],
						"tuned_search",
					)
				)
			),
		}
	)
	log_table_with_line_chart(
		run,
		"tuning/search_ranking",
		search_results_df[["rank_test_score", "mean_test_score"]],
		"rank_test_score",
		"mean_test_score",
		"RandomizedSearchCV Mean Test Score by Rank",
	)
	log_metrics_visuals(run, "tuning", y_test, tuned_pred, tuned_scores_pred)


def log_feature_importance_results(run, key, feature_importance_df, title):
	log_table_with_bar_chart(
		run,
		key,
		feature_importance_df.head(20),
		"feature",
		"importance",
		title,
	)


def log_benchmark_model_results(
	run,
	model_name,
	model,
	result,
	cv_scores,
	y_test,
	holdout_pred,
	holdout_scores,
	benchmark_feature_importance,
):
	cv_scores_df = pd.DataFrame(
		{
			"fold": np.arange(1, len(cv_scores) + 1),
			"f1": cv_scores,
		}
	)
	run.log({f"benchmark/{model_name}/{key}": value for key, value in result.items() if key != "model"})
	run.log(
		{
			f"benchmark/{model_name}/config": wandb.Table(
				dataframe=pd.DataFrame(build_model_config_rows(model_name, model, "benchmark"))
			),
		}
	)
	log_table_with_line_chart(
		run,
		f"benchmark/{model_name}/cv_folds",
		cv_scores_df,
		"fold",
		"f1",
		f"{model_name} CV F1 by Fold",
	)
	log_metrics_visuals(run, f"benchmark/{model_name}", y_test, holdout_pred, holdout_scores)
	if benchmark_feature_importance is not None:
		log_feature_importance_results(
			run,
			f"benchmark/{model_name}/feature_importance",
			benchmark_feature_importance,
			f"{model_name} Feature Importance (Top 20)",
		)


def log_benchmark_summary_results(run, benchmark_df):
	benchmark_summary_table = wandb.Table(dataframe=benchmark_df)
	run.log(
		{
			"benchmark/summary_table": benchmark_summary_table,
			"benchmark/holdout_f1_bar": wandb.plot.bar(
				benchmark_summary_table,
				"model",
				"holdout_f1",
				title="Benchmark Holdout F1",
			),
			"benchmark/holdout_accuracy_bar": wandb.plot.bar(
				benchmark_summary_table,
				"model",
				"holdout_accuracy",
				title="Benchmark Holdout Accuracy",
			),
			"benchmark/cv_f1_bar": wandb.plot.bar(
				benchmark_summary_table,
				"model",
				"cv_f1_mean",
				title="Benchmark CV Mean F1",
			),
		}
	)


def finish_wandb_run(run):
	run.finish()