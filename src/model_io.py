# ===============================
# model_io.py - MODEL IO HELPERS
# ===============================
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def save_model(model, model_type="lightgbm", fold=None):
	"""Save trained model."""
	MODEL_DIR.mkdir(parents=True, exist_ok=True)

	if fold is not None:
		filename = MODEL_DIR / f"{model_type}_fold_{fold}.pkl"
	else:
		filename = MODEL_DIR / f"{model_type}_final.pkl"

	with filename.open("wb") as file_handle:
		pickle.dump(model, file_handle)

	print(f"Model saved: {filename}")
	return str(filename)


def load_model(model_path):
	"""Load trained model for inference."""
	model_path = Path(model_path)
	with model_path.open("rb") as file_handle:
		model = pickle.load(file_handle)

	print(f"Model loaded: {model_path}")
	return model