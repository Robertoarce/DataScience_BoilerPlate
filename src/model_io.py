# ===============================
# model_io.py - MODEL IO HELPERS
# ===============================
import os
import pickle


def save_model(model, model_type="lightgbm", fold=None):
	"""Save trained model."""
	os.makedirs("models", exist_ok=True)

	if fold is not None:
		filename = f"models/{model_type}_fold_{fold}.pkl"
	else:
		filename = f"models/{model_type}_final.pkl"

	with open(filename, "wb") as file_handle:
		pickle.dump(model, file_handle)

	print(f"Model saved: {filename}")


def load_model(model_path):
	"""Load trained model for inference."""
	with open(model_path, "rb") as file_handle:
		model = pickle.load(file_handle)

	print(f"Model loaded: {model_path}")
	return model