from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from model_io import load_model
from wandb_utils import extract_prediction_scores


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_TEST_PATH = Path(r"C:\00 ALL\05 Kaggle\02 Titanic\test.csv")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "submissions" / "submission_ensemble.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
	parser = ArgumentParser(description="Generate Titanic predictions using an ensemble of saved models.")
	parser.add_argument(
		"--model-path",
		dest="model_paths",
		action="append",
		default=None,
		help="Path to a saved model pickle. Repeat this flag to pass multiple models.",
	)
	parser.add_argument(
		"--models-dir",
		default=str(DEFAULT_MODELS_DIR),
		help="Directory to auto-discover saved family-search models when --model-path is not provided.",
	)
	parser.add_argument(
		"--ensemble-type",
		default="soft_vote",
		choices=[
			"hard_vote",
			"soft_vote",
			"weighted_soft_vote",
			"weighted_hard_vote",
			"rank_average",
			"median_soft_vote",
		],
		help="How to combine predictions from multiple models.",
	)
	parser.add_argument(
		"--weights",
		default=None,
		help="Comma-separated weights for weighted ensembles, for example: 0.5,0.3,0.2",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.5,
		help="Decision threshold applied to the ensemble score.",
	)
	parser.add_argument(
		"--test-path",
		default=str(DEFAULT_TEST_PATH),
		help="Path to the Titanic test CSV.",
	)
	parser.add_argument(
		"--output-path",
		default=str(DEFAULT_OUTPUT_PATH),
		help="Path to write the submission CSV.",
	)
	return parser.parse_args()


def discover_model_paths(models_dir):
	models_dir = Path(models_dir)
	if not models_dir.exists():
		raise FileNotFoundError(f"Models directory not found: {models_dir}")

	family_models = sorted(models_dir.glob("family_search_*_final.pkl"))
	if family_models:
		return family_models

	best_model = models_dir / "best_family_search_final.pkl"
	if best_model.exists():
		return [best_model]

	raise FileNotFoundError(
		f"No saved ensemble candidates found in {models_dir}. "
		"Run training first so the saved model files are created."
	)


def parse_weights(raw_weights, n_models):
	if raw_weights is None:
		return np.ones(n_models) / n_models

	weights = np.array([float(value.strip()) for value in raw_weights.split(",")], dtype=float)
	if len(weights) != n_models:
		raise ValueError(f"Expected {n_models} weights, got {len(weights)}")
	if np.any(weights < 0):
		raise ValueError("Weights must be non-negative")
	if weights.sum() == 0:
		raise ValueError("Weights must not sum to zero")

	return weights / weights.sum()


def to_probability_like_scores(model, features):
	scores = extract_prediction_scores(model, features)

	if scores is None:
		predictions = model.predict(features)
		return np.asarray(predictions, dtype=float)

	scores = np.asarray(scores, dtype=float)

	if np.all((scores >= 0.0) & (scores <= 1.0)):
		return scores

	scores = np.clip(scores, -50, 50)
	return 1.0 / (1.0 + np.exp(-scores))


def load_models(model_paths):
	models = []
	for model_path in model_paths:
		model_path = Path(model_path)
		if not model_path.exists():
			raise FileNotFoundError(f"Saved model not found: {model_path}")
		models.append((model_path, load_model(model_path)))
	return models


def build_prediction_frame(models, test_df):
	rows = {"PassengerId": test_df["PassengerId"]}
	score_matrix = []
	prediction_matrix = []

	for model_path, model in models:
		model_name = model_path.stem
		scores = to_probability_like_scores(model, test_df)
		predictions = (scores >= 0.5).astype(int)

		rows[f"{model_name}_score"] = scores
		rows[f"{model_name}_pred"] = predictions
		score_matrix.append(scores)
		prediction_matrix.append(predictions)

	return pd.DataFrame(rows), np.vstack(score_matrix), np.vstack(prediction_matrix)


def ensemble_predictions(score_matrix, prediction_matrix, ensemble_type, weights, threshold):
	if ensemble_type == "hard_vote":
		votes = prediction_matrix.mean(axis=0)
		return (votes >= threshold).astype(int), votes

	if ensemble_type == "weighted_hard_vote":
		weighted_votes = np.average(prediction_matrix, axis=0, weights=weights)
		return (weighted_votes >= threshold).astype(int), weighted_votes

	if ensemble_type == "soft_vote":
		mean_scores = score_matrix.mean(axis=0)
		return (mean_scores >= threshold).astype(int), mean_scores

	if ensemble_type == "weighted_soft_vote":
		weighted_scores = np.average(score_matrix, axis=0, weights=weights)
		return (weighted_scores >= threshold).astype(int), weighted_scores

	if ensemble_type == "median_soft_vote":
		median_scores = np.median(score_matrix, axis=0)
		return (median_scores >= threshold).astype(int), median_scores

	if ensemble_type == "rank_average":
		rank_matrix = np.vstack(
			[
				pd.Series(scores).rank(method="average", pct=True).to_numpy()
				for scores in score_matrix
			]
		)
		rank_scores = rank_matrix.mean(axis=0)
		return (rank_scores >= threshold).astype(int), rank_scores

	raise ValueError(f"Unsupported ensemble type: {ensemble_type}")


def main():
	args = parse_args()
	test_path = Path(args.test_path)
	output_path = Path(args.output_path)

	if not test_path.exists():
		raise FileNotFoundError(f"Test dataset not found: {test_path}")

	test_df = pd.read_csv(test_path)
	if "PassengerId" not in test_df.columns:
		raise ValueError("The test dataset must include a PassengerId column.")

	model_paths = args.model_paths or [str(path) for path in discover_model_paths(args.models_dir)]
	models = load_models(model_paths)
	weights = parse_weights(args.weights, len(models))

	prediction_frame, score_matrix, prediction_matrix = build_prediction_frame(models, test_df)
	final_predictions, ensemble_scores = ensemble_predictions(
		score_matrix=score_matrix,
		prediction_matrix=prediction_matrix,
		ensemble_type=args.ensemble_type,
		weights=weights,
		threshold=args.threshold,
	)

	submission = pd.DataFrame(
		{
			"PassengerId": test_df["PassengerId"],
			"Survived": final_predictions.astype(int),
		}
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	submission.to_csv(output_path, index=False)

	debug_output_path = output_path.with_name(output_path.stem + "_details.csv")
	prediction_frame["ensemble_score"] = ensemble_scores
	prediction_frame["ensemble_pred"] = final_predictions
	prediction_frame.to_csv(debug_output_path, index=False)

	print("Loaded models:")
	for model_path, _ in models:
		print(f" - {model_path}")

	print(f"Ensemble type: {args.ensemble_type}")
	print(f"Submission saved: {output_path}")
	print(f"Detailed ensemble outputs saved: {debug_output_path}")
	print(submission.head().to_string(index=False))


if __name__ == "__main__":
	main()