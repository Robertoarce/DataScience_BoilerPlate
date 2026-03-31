# ===============================
# utils.py - UTILITY FUNCTIONS
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


def plot_feature_importance(model, feature_names=None, model_type="lightgbm", top_n=20):
    """Plot feature importance for trained model"""
    
    if model_type == "lightgbm":
        importance = model.feature_importances_
        importance_type = "gain"
    elif model_type == "catboost":
        importance = model.get_feature_importance()
        importance_type = "importance"
    
    # Create feature importance DataFrame
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
    
    feat_imp = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp.head(top_n), x='importance', y='feature')
    plt.title(f'{model_type.upper()} Feature Importance (Top {top_n})')
    plt.xlabel(f'Feature Importance ({importance_type})')
    plt.tight_layout()
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{model_type}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return feat_imp


def save_model(model, model_type="lightgbm", fold=None):
    """Save trained model"""
    os.makedirs("models", exist_ok=True)
    
    if fold is not None:
        filename = f"models/{model_type}_fold_{fold}.pkl"
    else:
        filename = f"models/{model_type}_final.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"📦 Model saved: {filename}")


def load_model(model_path):
    """Load trained model for inference"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"📥 Model loaded: {model_path}")
    return model