# 🏆 Kaggle Pro Repository Structure

A professional, production-ready template for Kaggle competitions with modular code organization, experiment tracking, and best practices.

## 📁 Project Structure

```
kaggle-project/
│
├── data/                    # Data storage
│   ├── raw/                 # Original, immutable data
│   ├── processed/           # Cleaned and preprocessed data
│   │   ├── features/        # Versioned feature sets
│   │   └── oof/             # Out-of-fold predictions for stacking
│   └── submissions/         # Competition submissions
│
├── src/                     # Source code
│   ├── config.py           # Configuration and hyperparameters (ENHANCED)
│   ├── features.py         # Feature engineering pipeline
│   ├── train.py            # Unified training with feature importance (ENHANCED)
│   ├── utils.py            # Utility functions (model persistence, plotting) (NEW)
│   ├── optuna_tuning.py    # Hyperparameter optimization (NEW)
│   ├── stacking.py         # Ensemble stacking methods (NEW)
│   ├── feature_versions.py # Feature versioning system (NEW)
│   └── ensemble_runner.py  # Complete pipeline orchestration (NEW)
│
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── logs/                   # Training logs and experiment artifacts
├── .gitignore             # Git ignore file
├── requirements.txt        # Python dependencies (ENHANCED)
└── README.md              # This file (ENHANCED)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create project directory
cd kaggle-project

# Install dependencies
pip install -r requirements.txt

# Setup Weights & Biases (optional)
wandb login
```

### 2. Prepare Data

Place your competition data in the appropriate folders:
- `data/raw/train.csv` - Training data
- `data/raw/test.csv` - Test data

### 3. Configure Project

Edit `src/config.py` to match your competition:
- Update `TARGET` column name
- Adjust model hyperparameters
- Set cross-validation strategy

### 4. Run Training Pipeline

#### Option A: Single Model Training
```bash
cd src

# Train LightGBM (default)
python train.py

# Train specific model
python train.py lightgbm
python train.py catboost

# Train both models
python train.py both
```

#### Option B: Full Ensemble Pipeline
```bash
cd src

# Run complete pipeline with all models and ensembles
python ensemble_runner.py
```

#### Option C: Hyperparameter Optimization
```bash
cd src

# Optimize hyperparameters first, then train
python optuna_tuning.py
```

This will:
- ✅ Load and preprocess data
- ✅ Run stratified K-fold cross-validation
- ✅ Train multiple models (LightGBM, CatBoost)
- ✅ Generate out-of-fold predictions for stacking
- ✅ Create stacked and weighted ensembles
- ✅ Generate multiple submission files
- ✅ Log experiments to Weights & Biases

## 🎯 Features

### 🔧 Modular Architecture
- **Separation of Concerns**: Each module has a single responsibility
- **Easy to Test**: Individual components can be tested in isolation
- **Reusable**: Code can be easily adapted for different competitions
- **Utility Functions**: Centralized plotting, model persistence, and analysis tools

### 🤖 Multiple Models & Ensembles
- **LightGBM**: Fast gradient boosting with early stopping
- **CatBoost**: Categorical feature handling without preprocessing
- **Stacking**: Meta-learning ensemble with cross-validation
- **Weighted Averaging**: Simple but effective ensemble method

### 📊 Experiment Tracking & Optimization
- **Weights & Biases Integration**: Automatic logging of metrics and artifacts
- **Feature Importance Analysis**: Automatic plotting and CSV export of feature importance
- **Model Persistence**: Save and load trained models for later inference
- **Optuna Hyperparameter Tuning**: Bayesian optimization for best parameters
- **Cross-Validation**: Robust model evaluation with stratified K-fold
- **OOF Predictions**: Out-of-fold predictions saved for ensemble stacking
- **Reproducibility**: Fixed random seeds and configuration management

### 🗂️ Advanced Data Management
- **Feature Versioning**: Save and load processed features to avoid recomputation
- **OOF Storage**: Automatic saving of out-of-fold predictions for stacking
- **Multiple Submissions**: Generate submissions for each model and ensemble

### 🏗️ Production-Ready
- **Type Hints**: Better code documentation and IDE support
- **Error Handling**: Graceful error management and logging
- **Configuration Management**: Centralized hyperparameters and settings
- **Ensemble Pipeline**: Complete automation from training to final submission

## 🔬 Advanced Features & Workflow

### 🎯 Hyperparameter Optimization with Optuna

```python
# Optimize LightGBM hyperparameters
from optuna_tuning import run_hyperparameter_tuning
from features import build_features
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
X, y, X_test = build_features(train, test)

best_params = run_hyperparameter_tuning(X, y, n_trials=100)
```

### 🥞 Ensemble Stacking

```python
# After training base models, create stacked ensemble
from stacking import run_stacking

# This uses saved OOF predictions from both models
stacking_score = run_stacking()
```

### 💾 Feature Versioning

```python
# Save processed features to avoid recomputation
from feature_versions import save_processed_features, load_processed_features

# Save features
save_processed_features(X_train, X_test, y_train, version="v2")

# Load features
X_train, X_test, y_train, feature_names = load_processed_features(version="v2")
```

### 🏃‍♂️ Complete Automated Pipeline

```python
# Run everything automatically
from ensemble_runner import EnsembleRunner

runner = EnsembleRunner()
results = runner.run_full_pipeline(
    optimize_first=True,  # Run Optuna optimization first
    n_trials=50
)
```

### 🛠️ Utility Functions

```python
# Feature importance analysis and visualization
from utils import plot_feature_importance, save_model, load_model

# Plot feature importance for any trained model
feat_imp = plot_feature_importance(model, feature_names, model_type="lightgbm")

# Save trained models for later use
save_model(model, model_type="lightgbm")

# Load saved models for inference
loaded_model = load_model("models/lightgbm_final.pkl")
predictions = loaded_model.predict_proba(X_test)[:, 1]
```

## 📝 Usage Examples

### Custom Feature Engineering

```python
# Edit src/features.py
def build_features(train, test):
    # Add your custom features here
    train['custom_feature'] = train['col1'] * train['col2']
    test['custom_feature'] = test['col1'] * test['col2']
    
    # Continue with preprocessing pipeline...
```

### Custom Model Parameters

```python
# Edit src/config.py
class Config:
    LGB_PARAMS = {
        "learning_rate": 0.1,  # Faster learning
        "num_leaves": 64,      # More complex trees
        # ... other parameters
    }
```

### Custom Metrics

```python
# Edit src/train.py
from sklearn.metrics import f1_score

# Replace roc_auc_score with your metric
final_score = f1_score(y, oof > 0.5)
```

## 🏅 Best Practices

### ✅ Data Management
- Keep raw data immutable
- Version your processed datasets
- Document data transformations

### ✅ Experiment Tracking
- Log all hyperparameters and metrics
- Track model artifacts and predictions
- Document experiment insights

### ✅ Code Quality
- Use consistent naming conventions
- Write docstrings for complex functions
- Keep functions small and focused

### ✅ Model Development
- Always use cross-validation
- Implement early stopping
- Monitor for overfitting

## 🔧 Customization

This template is designed to be easily customizable for different competition types:

- **Binary Classification**: Default setup
- **Multi-class**: Update target encoding and metrics
- **Regression**: Change model type and evaluation metrics
- **Time Series**: Modify cross-validation strategy

## 📊 Competition Workflow

1. **EDA Phase**: Use notebooks for exploratory data analysis and insights
2. **Feature Development**: Implement feature engineering in `src/features.py`
3. **Model Configuration**: Adjust model parameters in `src/config.py`
4. **Hyperparameter Tuning**: Run `src/optuna_tuning.py` for optimization
5. **Final Training**: Execute full pipeline with `src/train.py` or `src/ensemble_runner.py`
6. **Submission**: Use generated files from `data/submissions/`

## 🤝 Contributing

Feel free to extend this template with:
- Additional model types (XGBoost, CatBoost, Neural Networks)
- Advanced feature engineering techniques
- Hyperparameter optimization (Optuna integration)
- Ensemble methods
- Custom validation strategies

## 📄 License

This template is free to use for any Kaggle competition or machine learning project.

---

**Happy Kaggling! 🎯**