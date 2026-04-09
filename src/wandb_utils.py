import os
import sys
import tempfile

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

try:
	import shap
except ImportError:
	shap = None

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, roc_curve
from sklearn.tree import export_graphviz


CLASS_NAMES = ["Not Survived", "Survived"]


def ensure_graphviz_path():
	dot_dir = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
	if os.path.exists(os.path.join(dot_dir, "dot.exe")) and dot_dir not in os.environ.get("PATH", ""):
		os.environ["PATH"] = dot_dir + os.pathsep + os.environ.get("PATH", "")
	return os.path.exists(os.path.join(dot_dir, "dot.exe"))


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


def build_permutation_importance_frame(
	model_name,
	estimator,
	features,
	target,
	scoring="f1",
	n_repeats=10,
	random_state=42,
	feature_names=None,
):
	importance = permutation_importance(
		estimator,
		features,
		target,
		scoring=scoring,
		n_repeats=n_repeats,
		random_state=random_state,
		n_jobs=-1,
	)

	if feature_names is not None:
		resolved_feature_names = list(feature_names)
	elif hasattr(features, "columns"):
		feature_names = list(features.columns)
	else:
		feature_names = [f"feature_{index}" for index in range(len(importance.importances_mean))]
		resolved_feature_names = feature_names

	if feature_names is not None and "resolved_feature_names" not in locals():
		resolved_feature_names = feature_names

	return (
		pd.DataFrame(
			{
				"model": model_name,
				"feature": resolved_feature_names,
				"importance_mean": importance.importances_mean,
				"importance_std": importance.importances_std,
			}
		)
		.sort_values(by="importance_mean", ascending=False)
		.reset_index(drop=True)
	)


def build_shap_artifacts(
	model_name,
	estimator,
	background_features,
	evaluation_features,
	random_state=42,
	max_background_samples=200,
	max_evaluation_samples=200,
):
	if shap is None:
		return None

	background_sample = background_features.sample(
		n=min(max_background_samples, len(background_features)),
		random_state=random_state,
	)
	evaluation_sample = evaluation_features.sample(
		n=min(max_evaluation_samples, len(evaluation_features)),
		random_state=random_state,
	)

	try:
		explainer = shap.Explainer(estimator, background_sample)
		shap_values = explainer(evaluation_sample)
	except Exception:
		return None

	values = shap_values.values
	if values.ndim == 3:
		values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

	if values.ndim != 2:
		return None

	shap_summary_df = (
		pd.DataFrame(
			{
				"model": model_name,
				"feature": list(evaluation_sample.columns),
				"mean_abs_shap": np.abs(values).mean(axis=0),
			}
		)
		.sort_values(by="mean_abs_shap", ascending=False)
		.reset_index(drop=True)
	)

	return {
		"summary_df": shap_summary_df,
		"values": values,
		"evaluation_sample": evaluation_sample,
	}


def build_tree_graphviz_artifacts(model_name, estimator, feature_names, max_depth=3):
	if not ensure_graphviz_path():
		return None

	graph_source = None
	title = None

	if hasattr(estimator, "tree_"):
		graph_source = export_graphviz(
			estimator,
			out_file=None,
			feature_names=list(feature_names),
			class_names=CLASS_NAMES,
			filled=True,
			rounded=True,
			special_characters=True,
			max_depth=max_depth,
		)
		title = f"{model_name} tree"
	elif hasattr(estimator, "estimators_") and len(estimator.estimators_) > 0:
		first_tree = estimator.estimators_[0]
		if isinstance(first_tree, np.ndarray):
			first_tree = first_tree.ravel()[0]
		graph_source = export_graphviz(
			first_tree,
			out_file=None,
			feature_names=list(feature_names),
			class_names=CLASS_NAMES,
			filled=True,
			rounded=True,
			special_characters=True,
			max_depth=max_depth,
		)
		title = f"{model_name} first tree"
	elif hasattr(estimator, "get_booster"):
		try:
			from xgboost import to_graphviz
			xgb_graph = to_graphviz(estimator, num_trees=0)
			graph_source = xgb_graph.source
			title = f"{model_name} first boosted tree"
		except Exception:
			return None

	if graph_source is None:
		return None

	try:
		png_bytes = graphviz.Source(graph_source).pipe(format="png")
	except Exception:
		return None

	return {
		"title": title,
		"dot_source": graph_source,
		"png_bytes": png_bytes,
	}


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


def log_permutation_importance_results(run, key, permutation_importance_df, title):
	log_table_with_bar_chart(
		run,
		key,
		permutation_importance_df.head(20),
		"feature",
		"importance_mean",
		title,
	)


def log_shap_results(run, key, shap_artifacts, title):
	if shap_artifacts is None:
		return

	log_table_with_bar_chart(
		run,
		key,
		shap_artifacts["summary_df"].head(20),
		"feature",
		"mean_abs_shap",
		title,
	)

	if shap is None:
		return

	try:
		shap.summary_plot(
			shap_artifacts["values"],
			shap_artifacts["evaluation_sample"],
			show=False,
			max_display=20,
		)
		figure = plt.gcf()
		run.log({f"{key}/beeswarm": wandb.Image(figure)})
		plt.close(figure)
	except Exception:
		plt.close("all")


def log_tree_graphviz_results(run, key, tree_graphviz_artifacts):
	if tree_graphviz_artifacts is None:
		return

	with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
		temp_file.write(tree_graphviz_artifacts["png_bytes"])
		temp_path = temp_file.name

	try:
		run.log({f"{key}/graphviz": wandb.Image(temp_path, caption=tree_graphviz_artifacts["title"])})
		run.log(
			{
				f"{key}/dot_source": wandb.Table(
					dataframe=pd.DataFrame(
						[{"title": tree_graphviz_artifacts["title"], "dot_source": tree_graphviz_artifacts["dot_source"]}]
					)
				)
			}
		)
	finally:
		try:
			os.unlink(temp_path)
		except OSError:
			pass


def log_model_search_results(
	run,
	key,
	search,
	result,
	y_test,
	y_pred,
	y_scores,
	feature_importance_df=None,
	permutation_importance_df=None,
	transformed_permutation_importance_df=None,
	shap_artifacts=None,
	tree_graphviz_artifacts=None,
):
	best_params_df = pd.DataFrame(
		[
			{"parameter": parameter, "value": str(value)}
			for parameter, value in sorted(search.best_params_.items())
		]
	)

	run.log({f"{key}/{metric_name}": metric_value for metric_name, metric_value in result.items() if metric_name != "model"})
	run.log({f"{key}/best_params": wandb.Table(dataframe=best_params_df)})
	log_metrics_visuals(run, key, y_test, y_pred, y_scores)

	if feature_importance_df is not None:
		log_feature_importance_results(run, f"{key}/feature_importance", feature_importance_df, f"{key} Feature Importance (Top 20)")

	if permutation_importance_df is not None:
		log_permutation_importance_results(
			run,
			f"{key}/permutation_importance",
			permutation_importance_df,
			f"{key} Permutation Importance (Top 20)",
		)

	if transformed_permutation_importance_df is not None:
		log_permutation_importance_results(
			run,
			f"{key}/permutation_importance_transformed",
			transformed_permutation_importance_df,
			f"{key} Transformed Permutation Importance (Top 20)",
		)

	if shap_artifacts is not None:
		log_shap_results(run, f"{key}/shap", shap_artifacts, f"{key} SHAP Mean Absolute Impact (Top 20)")

	if tree_graphviz_artifacts is not None:
		log_tree_graphviz_results(run, f"{key}/tree", tree_graphviz_artifacts)


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
	benchmark_permutation_importance,
	benchmark_tree_graphviz_artifacts=None,
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
	if benchmark_permutation_importance is not None:
		log_permutation_importance_results(
			run,
			f"benchmark/{model_name}/permutation_importance",
			benchmark_permutation_importance,
			f"{model_name} Permutation Importance (Top 20)",
		)
	if benchmark_tree_graphviz_artifacts is not None:
		log_tree_graphviz_results(run, f"benchmark/{model_name}/tree", benchmark_tree_graphviz_artifacts)


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


def log_search_comparison_results(run, key, results_df, title_prefix):
	comparison_table = wandb.Table(dataframe=results_df)
	run.log(
		{
			f"{key}/summary_table": comparison_table,
			f"{key}/holdout_f1_bar": wandb.plot.bar(
				comparison_table,
				"model",
				"holdout_f1",
				title=f"{title_prefix} Holdout F1",
			),
			f"{key}/holdout_accuracy_bar": wandb.plot.bar(
				comparison_table,
				"model",
				"holdout_accuracy",
				title=f"{title_prefix} Holdout Accuracy",
			),
			f"{key}/best_cv_f1_bar": wandb.plot.bar(
				comparison_table,
				"model",
				"best_cv_f1",
				title=f"{title_prefix} Best CV F1",
			),
		}
	)


def log_model_artifact(run, model_path, artifact_name, metadata=None, aliases=None):
	if run is None or not os.path.exists(model_path):
		return

	artifact = wandb.Artifact(
		artifact_name,
		type="model",
		metadata=serialize_for_wandb(metadata or {}),
	)
	artifact.add_file(model_path, name=os.path.basename(model_path))
	run.log_artifact(artifact, aliases=aliases or ["latest"])


def finish_wandb_run(run):
	run.finish()