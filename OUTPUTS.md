# Kaggle Pro Repository - Generated Outputs

When you run the training pipeline, the following files and folders will be automatically created:

## 📁 Generated Structure:
```
kaggle-project/
├── data/
│   ├── processed/
│   │   └── oof/
│   │       ├── lgb_oof.npy       # LightGBM out-of-fold predictions
│   │       ├── lgb_test.npy      # LightGBM test predictions
│   │       ├── cb_oof.npy        # CatBoost out-of-fold predictions
│   │       └── cb_test.npy       # CatBoost test predictions
│   └── submissions/
│       ├── submission.csv        # LightGBM submission
│       ├── catboost_submission.csv # CatBoost submission
│       ├── stacked_submission.csv # Stacking ensemble
│       └── weighted_ensemble_submission.csv
├── models/
│   ├── lightgbm_final.pkl       # Saved LightGBM model
│   └── catboost_final.pkl        # Saved CatBoost model
└── plots/
    ├── lightgbm_feature_importance.png
    ├── lightgbm_feature_importance.csv
    ├── catboost_feature_importance.png
    └── catboost_feature_importance.csv
```

## 🎯 Key Features Added:
- **Feature Importance Plots**: Automatic visualization of most important features
- **Model Persistence**: Trained models saved as .pkl files for later use
- **CSV Exports**: Feature importance rankings saved as spreadsheets
- **W&B Integration**: Feature importance logged to Weights & Biases

## 📊 Usage Examples:

**Load a saved model:**
```python
from utils import load_model
model = load_model("models/lightgbm_final.pkl")
predictions = model.predict_proba(X_test)[:, 1]
```

**Generate feature importance plots:**
```python
from utils import plot_feature_importance
feat_imp = plot_feature_importance(model, feature_names, model_type="lightgbm", top_n=20)
```

**Save models during training:**
```python
from utils import save_model
save_model(model, model_type="lightgbm", fold=0)  # Save fold-specific model
save_model(model, model_type="lightgbm")          # Save final model
```

**View feature importance:**
- Check the `plots/` directory for PNG visualizations
- Open CSV files to see ranked feature importance scores
- View interactive plots in your W&B dashboard