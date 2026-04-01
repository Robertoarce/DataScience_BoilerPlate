# Feature Versioning Guide

This guide shows how to create, manage, and use different versions of features in your machine learning pipeline.

## 🎯 Quick Start

```bash
# Create a new feature version
python src/feature_manager.py create v2

# Train a model with specific features
python src/train.py lightgbm --features v2

# Compare feature versions
python src/feature_manager.py compare v1 v2

# List all versions
python src/feature_manager.py list
```

## 📁 Feature Versions Available

### v1 - Basic Features
- **What**: Basic preprocessing with median imputation and one-hot encoding
- **When to use**: Baseline model, simple datasets
- **Speed**: ⚡ Fast
- **Features**: ~Original feature count

### v2 - Enhanced Features  
- **What**: Interaction features, scaling, controlled one-hot encoding
- **When to use**: Better performance needed, medium datasets
- **Speed**: ⚡⚡ Moderate  
- **Features**: ~2-3x original feature count

### v3 - Advanced Features
- **What**: Statistical features + feature selection (top 500 features)
- **When to use**: Complex patterns, large datasets, competition
- **Speed**: ⚡⚡⚡ Slower
- **Features**: Up to 500 selected features

## 🔧 Command Reference

### Feature Management

```bash
# Create feature versions
python src/feature_manager.py create v1    # Basic
python src/feature_manager.py create v2    # Enhanced  
python src/feature_manager.py create v3    # Advanced

# Overwrite existing version
python src/feature_manager.py create v2 --force

# List all versions with metadata
python src/feature_manager.py list

# Compare two versions  
python src/feature_manager.py compare v1 v2

# Benchmark all versions (quick LogisticRegression test)
python src/feature_manager.py benchmark

# Clean old versions (keep latest 3)
python src/feature_manager.py clean --keep 3
```

### Model Training with Versions

```bash
# Train with specific feature version
python src/train.py lightgbm --features v2
python src/train.py catboost --features v3
python src/train.py xgboost --features v1

# Train all models with same features
python src/train.py all --features v3

# Disable feature caching (rebuild every time)
python src/train.py lightgbm --features v2 --no-cache

# Combine with other options
python src/train.py lightgbm --features v3 --trials 50 --no-optuna
```

### Advanced Usage

```bash
# Quick experimentation workflow
python src/feature_manager.py create v4 --force
python src/train.py lightgbm --features v4 --trials 10
python src/feature_manager.py compare v3 v4

# Production training workflow  
python src/feature_manager.py benchmark  # Find best version
python src/train.py all --features v3 --trials 100  # Train all models
python src/ensemble_runner.py  # Create ensembles
```

## 📊 File Structure

Feature versions are stored in `data/processed/features/`:

```
data/processed/features/
├── v1/
│   ├── X_train.npy         # Training features
│   ├── X_test.npy          # Test features
│   ├── y_train.npy         # Target values
│   ├── feature_names.csv   # Feature names (if available)
│   └── metadata.csv        # Version metadata
├── v2/
│   └── ...
└── v3/
    └── ...
```

## 🔄 Typical Workflow

### 1. Initial Setup
```bash
# Start with basic features
python src/feature_manager.py create v1
python src/train.py lightgbm --features v1
```

### 2. Feature Iteration
```bash
# Create enhanced features
python src/feature_manager.py create v2

# Compare versions
python src/feature_manager.py compare v1 v2

# Test new features
python src/train.py lightgbm --features v2 --trials 20
```

### 3. Production Training
```bash
# Find best version
python src/feature_manager.py benchmark

# Train all models with best features
python src/train.py all --features v3 --trials 100

# Create ensembles
python src/ensemble_runner.py
```

## 💡 Tips & Best Practices

### Performance Tips
- **v1** is fastest for quick experiments
- **v3** usually gives best performance but takes longer
- Use `--no-cache` only when debugging feature engineering

### Storage Management
- Each version stores ~2-4 files per feature set
- Use `feature_manager.py clean` to remove old versions
- Typical version takes 10-100MB depending on dataset size

### Experimentation
- Always run `benchmark` to quickly compare versions
- Use `compare` to understand feature differences
- Start with v1, then try v2, then v3 if needed

### Integration with Ensemble
- The ensemble system automatically looks for versioned predictions
- Files are saved as: `lgb_v2_oof.npy`, `cb_v3_test.npy`, etc.
- Make sure all base models use the same feature version for ensembling

## 🐛 Troubleshooting

### "Feature version doesn't exist"
```bash
python src/feature_manager.py list  # Check available versions
python src/feature_manager.py create v2  # Create missing version
```

### "Features already exist"
```bash
python src/feature_manager.py create v2 --force  # Overwrite
```

### "Out of memory during feature creation"
- Try v1 (simplest features)
- Reduce dataset size for testing
- Check available RAM

### "Different feature shapes in ensemble"
- Ensure all models use same feature version
- Check that features were built with same raw data

## 🚀 Advanced: Creating Custom Versions

To create your own feature version (e.g., v4), edit `src/features.py`:

```python
def build_features_v4(train, test):
    """Version 4: Your custom features"""
    X = train.drop(columns=["target"])
    y = train["target"]
    
    # Your feature engineering here
    # ...
    
    return X_processed, y, test_processed, feature_names

# Add to build_features() function
elif version == "v4":
    X_processed, y, test_processed, feature_names = build_features_v4(train, test)
```

Then use it:
```bash
python src/feature_manager.py create v4
python src/train.py lightgbm --features v4
```

## 📈 Example Results

Typical performance gains:
- **v1 → v2**: +0.001 to +0.005 AUC improvement
- **v2 → v3**: +0.001 to +0.003 AUC improvement  
- **v1 → v3**: +0.002 to +0.008 AUC improvement

Remember: Results vary by dataset. Always benchmark on your data!