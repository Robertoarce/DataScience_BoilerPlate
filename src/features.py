# ===============================
# features.py - Enhanced Feature Engineering
# ===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from feature_versions import save_features, load_features, features_exist


def build_features_v1(train, test):
    """Version 1: Basic preprocessing (original)"""
    X = train.drop(columns=["target"])
    y = train["target"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)

    return X_processed, y, test_processed


def build_features_v2(train, test):
    """Version 2: Enhanced with scaling and interaction features"""
    X = train.drop(columns=["target"])
    y = train["target"]

    # Separate column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # Create interaction features for numerical columns
    interaction_features = []
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i+1:]:
            # Multiplicative interactions
            interaction_name = f"{col1}_x_{col2}"
            X[interaction_name] = X[col1] * X[col2]
            test[interaction_name] = test[col1] * test[col2]
            interaction_features.append(interaction_name)
            
            # Ratio features (avoid division by zero)
            ratio_name = f"{col1}_div_{col2}"
            X[ratio_name] = X[col1] / (X[col2] + 1e-8)
            test[ratio_name] = test[col1] / (test[col2] + 1e-8)
            interaction_features.append(ratio_name)

    # Update numerical columns list
    all_num_cols = list(num_cols) + interaction_features

    # Enhanced preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), all_num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=50))
        ]), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)

    # Feature names for later use
    num_feature_names = all_num_cols
    cat_feature_names = list(preprocessor.named_transformers_['cat']
                            .named_steps['ohe'].get_feature_names_out(cat_cols))
    feature_names = num_feature_names + cat_feature_names

    return X_processed, y, test_processed, feature_names


def build_features_v3(train, test):
    """Version 3: Advanced with statistical features and feature selection"""
    X = train.drop(columns=["target"])
    y = train["target"]

    # Get basic features first
    X_enhanced, _, test_enhanced, base_feature_names = build_features_v2(train, test)

    # Add statistical features
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    
    # Row-wise statistics
    X_stats = pd.DataFrame()
    test_stats = pd.DataFrame()
    
    X_stats['row_sum'] = X[num_cols].sum(axis=1)
    X_stats['row_mean'] = X[num_cols].mean(axis=1)
    X_stats['row_std'] = X[num_cols].std(axis=1)
    X_stats['row_max'] = X[num_cols].max(axis=1)
    X_stats['row_min'] = X[num_cols].min(axis=1)
    X_stats['row_median'] = X[num_cols].median(axis=1)
    
    test_stats['row_sum'] = test[num_cols].sum(axis=1)
    test_stats['row_mean'] = test[num_cols].mean(axis=1)
    test_stats['row_std'] = test[num_cols].std(axis=1)
    test_stats['row_max'] = test[num_cols].max(axis=1)
    test_stats['row_min'] = test[num_cols].min(axis=1)
    test_stats['row_median'] = test[num_cols].median(axis=1)

    # Combine features
    X_combined = np.hstack([X_enhanced, X_stats.values])
    test_combined = np.hstack([test_enhanced, X_stats.values])
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(500, X_combined.shape[1]))  # Keep top 500 features
    X_selected = selector.fit_transform(X_combined, y)
    test_selected = selector.transform(test_combined)
    
    # Get selected feature names
    stat_feature_names = X_stats.columns.tolist()
    all_feature_names = base_feature_names + stat_feature_names
    selected_features = selector.get_support(indices=True)
    selected_feature_names = [all_feature_names[i] for i in selected_features]

    return X_selected, y, test_selected, selected_feature_names


def build_features_v4(train, test):
    """Version 4: Adaptive/Dynamic feature engineering"""
    from dynamic_features import DynamicFeatureEngineer
    
    print("🧠 Using adaptive feature engineering...")
    engineer = DynamicFeatureEngineer()
    X_processed, y, test_processed, feature_names = engineer.build_features_adaptive(
        train, test,
        create_interactions=True,
        create_stats=True, 
        apply_selection=True
    )
    
    return X_processed, y, test_processed, feature_names


def build_features(train, test, version="v1", use_cache=True):
    """
    Main feature building function with versioning support
    
    Args:
        train: Training dataframe
        test: Test dataframe  
        version: Feature version ("v1", "v2", "v3", "v4")
        use_cache: Whether to use cached features if available
    
    Returns:
        X_processed, y, test_processed, feature_names (if available)
    """
    
    # Check if cached features exist
    if use_cache and features_exist(version):
        print(f"📁 Loading cached features version {version}...")
        X_processed, test_processed, y, feature_names = load_features(version)
        return X_processed, y, test_processed, feature_names
    
    print(f"🔨 Building features version {version}...")
    
    # Build features based on version
    if version == "v1":
        X_processed, y, test_processed = build_features_v1(train, test)
        feature_names = None
    elif version == "v2":
        X_processed, y, test_processed, feature_names = build_features_v2(train, test)
    elif version == "v3":
        X_processed, y, test_processed, feature_names = build_features_v3(train, test)
    elif version == "v4":
        X_processed, y, test_processed, feature_names = build_features_v4(train, test)
    else:
        raise ValueError(f"Unknown feature version: {version}. Available: v1, v2, v3, v4")
    
    # Save features for future use
    if use_cache:
        print(f"💾 Saving features version {version}...")
        save_features(X_processed, test_processed, y, feature_names, version)
    
    return X_processed, y, test_processed, feature_names


# Legacy function for backward compatibility
def build_features_legacy(train, test):
    """Legacy function - calls v1 without caching"""
    X_processed, y, test_processed = build_features_v1(train, test)
    return X_processed, y, test_processed