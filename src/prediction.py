from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from model_io import load_model


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_family_search_final.pkl"
DEFAULT_TEST_PATH = Path(r"C:\00 ALL\05 Kaggle\02 Titanic\test.csv")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "submissions" / "submission_best_family_search.csv"


def parse_args():
	parser = ArgumentParser(description="Load a saved Titanic model and generate predictions.")
	parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the saved model pickle file.")
	parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH), help="Path to the Titanic test dataset CSV.")
	parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Path to write the submission CSV.")
	return parser.parse_args()


def main():
	args = parse_args()
	model_path = Path(args.model_path)
	test_path = Path(args.test_path)
	output_path = Path(args.output_path)

	if not model_path.exists():
		raise FileNotFoundError(f"Saved model not found: {model_path}")

	if not test_path.exists():
		raise FileNotFoundError(f"Test dataset not found: {test_path}")

	test_df = pd.read_csv(test_path)
	if "PassengerId" not in test_df.columns:
		raise ValueError("The test dataset must include a PassengerId column.")

	model = load_model(model_path)
	predictions = model.predict(test_df)

	submission = pd.DataFrame(
		{
			"PassengerId": test_df["PassengerId"],
			"Survived": predictions.astype(int),
		}
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	submission.to_csv(output_path, index=False)

	print(f"Predictions saved: {output_path}")
	print(submission.head().to_string(index=False))


if __name__ == "__main__":
	main()