# ===============================
# dynamic_features.py - Data-Driven Feature Engineering
# ===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class DynamicFeatureEngineer:
    """Data-driven feature engineering that adapts to your dataset"""
    
    def __init__(self, target_col="target", random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.feature_importance_scores = {}
        
    def analyze_data(self, df):
        """Analyze dataset characteristics to inform feature engineering strategy"""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        analysis = {
            'n_samples': len(df),
            'n_features': len(X.columns),
            'numerical_cols': list(X.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_cols': list(X.select_dtypes(include=['object', 'category']).columns),
            'missing_percentage': (X.isnull().sum() / len(X) * 100).to_dict(),
            'high_cardinality_cats': [],
            'low_variance_nums': [],
            'target_correlation': {}
        }
        
        # Identify high cardinality categorical features
        for col in analysis['categorical_cols']:
            if X[col].nunique() > 50:  # Threshold for high cardinality
                analysis['high_cardinality_cats'].append(col)
        
        # Identify low variance numerical features
        for col in analysis['numerical_cols']:
            if X[col].var() < 0.01:  # Low variance threshold
                analysis['low_variance_nums'].append(col)
        
        # Calculate target correlations for numerical features
        for col in analysis['numerical_cols']:
            try:
                corr = np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1]
                analysis['target_correlation'][col] = corr
            except:
                analysis['target_correlation'][col] = 0
                
        return analysis
    
    def create_interaction_features(self, X, y, max_features=50, correlation_threshold=0.1):
        """Dynamically create interactions based on feature importance"""
        
        # Get top correlated numerical features
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        correlations = {}
        
        for col in num_cols:
            try:
                corr = abs(np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1])
                correlations[col] = corr
            except:
                correlations[col] = 0
        
        # Select top correlated features for interactions
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [col for col, corr in top_features[:10] if corr > correlation_threshold]
        
        print(f"🔗 Creating interactions from {len(selected_features)} top features: {selected_features}")
        
        # Create interactions only for selected features
        interaction_features = []
        for i, col1 in enumerate(selected_features):
            for col2 in selected_features[i+1:]:
                if len(interaction_features) >= max_features:
                    break
                    
                # Multiplicative interaction
                interaction_name = f"{col1}_x_{col2}"
                X[interaction_name] = X[col1] * X[col2]
                interaction_features.append(interaction_name)
                
                # Ratio interaction (if both features are positive on average)
                if X[col1].mean() > 0 and X[col2].mean() > 0:
                    ratio_name = f"{col1}_div_{col2}"
                    X[ratio_name] = X[col1] / (X[col2] + 1e-8)
                    interaction_features.append(ratio_name)
            
            if len(interaction_features) >= max_features:
                break
        
        print(f"✅ Created {len(interaction_features)} interaction features")
        return X, interaction_features
    
    def create_statistical_features(self, X, min_numeric_features=3):
        """Create statistical features only if we have enough numerical columns"""
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        if len(num_cols) < min_numeric_features:
            print(f"⏭️ Skipping statistical features (only {len(num_cols)} numeric columns)")
            return X, []
            
        print(f"📊 Creating statistical features from {len(num_cols)} numerical columns")
        
        # Row-wise statistics
        stat_features = []
        X_numeric = X[num_cols]
        
        # Basic statistics
        X['row_sum'] = X_numeric.sum(axis=1)
        X['row_mean'] = X_numeric.mean(axis=1)
        X['row_std'] = X_numeric.std(axis=1)
        X['row_max'] = X_numeric.max(axis=1)
        X['row_min'] = X_numeric.min(axis=1)
        X['row_median'] = X_numeric.median(axis=1)
        
        stat_features = ['row_sum', 'row_mean', 'row_std', 'row_max', 'row_min', 'row_median']
        
        # Advanced statistics (only if we have many features)
        if len(num_cols) >= 5:
            X['row_skew'] = X_numeric.skew(axis=1)
            X['row_kurtosis'] = X_numeric.kurtosis(axis=1)
            X['row_q25'] = X_numeric.quantile(0.25, axis=1)
            X['row_q75'] = X_numeric.quantile(0.75, axis=1)
            X['row_iqr'] = X['row_q75'] - X['row_q25']
            
            stat_features.extend(['row_skew', 'row_kurtosis', 'row_q25', 'row_q75', 'row_iqr'])
        
        print(f"✅ Created {len(stat_features)} statistical features")
        return X, stat_features
    
    def smart_preprocessing(self, X, analysis):
        """Choose preprocessing strategy based on data characteristics"""
        
        num_cols = analysis['numerical_cols']
        cat_cols = analysis['categorical_cols']
        
        # Handle numerical features
        if analysis['n_samples'] > 10000:
            # Large dataset: use robust scaling
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        else:
            # Small dataset: simpler imputation
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])
        
        # Handle categorical features based on cardinality
        cat_transformers = []
        
        for col in cat_cols:
            unique_values = X[col].nunique()
            
            if unique_values <= 10:
                # Low cardinality: standard one-hot encoding
                transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])
            elif unique_values <= 50:
                # Medium cardinality: limited one-hot encoding
                transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', max_categories=20))
                ])
            else:
                # High cardinality: target encoding or simple label encoding
                # For now, use limited one-hot to avoid too many features
                transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', max_categories=10))
                ])
            
            cat_transformers.append((f'cat_{col}', transformer, [col]))
        
        # Combine transformers
        transformers = [('num', num_pipeline, num_cols)]
        transformers.extend(cat_transformers)
        
        preprocessor = ColumnTransformer(transformers)
        return preprocessor
    
    def adaptive_feature_selection(self, X, y, target_features=None):
        """Choose feature selection strategy based on dataset size and features"""
        
        n_samples, n_features = X.shape
        
        if target_features is None:
            # Adaptive target based on dataset size
            if n_samples < 1000:
                target_features = min(50, n_features // 2)
            elif n_samples < 10000:
                target_features = min(200, n_features // 2)
            else:
                target_features = min(500, n_features // 2)
        
        print(f"🎯 Selecting {target_features} features from {n_features} available")
        
        # Choose selection method based on dataset characteristics
        if n_features > 1000:
            # Many features: use fast univariate selection
            selector = SelectKBest(f_classif, k=target_features)
        else:
            # Fewer features: could use more sophisticated selection
            selector = SelectKBest(mutual_info_classif, k=target_features)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get feature importance scores
        feature_scores = dict(zip(range(len(selector.scores_)), selector.scores_))
        selected_indices = selector.get_support(indices=True)
        
        return X_selected, selected_indices, feature_scores
    
    def build_features_adaptive(self, train, test, create_interactions=True, 
                              create_stats=True, apply_selection=True):
        """Main adaptive feature engineering pipeline"""
        
        print("🧠 Analyzing data characteristics...")
        analysis = self.analyze_data(train)
        
        # Print analysis summary
        print(f"📊 Dataset: {analysis['n_samples']} samples, {analysis['n_features']} features")
        print(f"   Numerical: {len(analysis['numerical_cols'])}")
        print(f"   Categorical: {len(analysis['categorical_cols'])}")
        print(f"   High cardinality cats: {len(analysis['high_cardinality_cats'])}")
        
        X = train.drop(columns=[self.target_col]).copy()
        y = train[self.target_col]
        X_test = test.copy()
        
        # Create derived features based on data characteristics
        if create_interactions and len(analysis['numerical_cols']) >= 2:
            print("\n🔗 Creating interaction features...")
            X, interaction_features = self.create_interaction_features(X, y)
            
            # Apply same transformations to test set
            for feature in interaction_features:
                if '_x_' in feature:
                    col1, col2 = feature.replace('_x_', '|').split('|')
                    X_test[feature] = X_test[col1] * X_test[col2]
                elif '_div_' in feature:
                    col1, col2 = feature.replace('_div_', '|').split('|')
                    X_test[feature] = X_test[col1] / (X_test[col2] + 1e-8)
        
        if create_stats and len(analysis['numerical_cols']) >= 3:
            print("\n📊 Creating statistical features...")
            X, stat_features = self.create_statistical_features(X)
            
            # Apply same transformations to test set
            num_cols = [col for col in analysis['numerical_cols'] if col in X_test.columns]
            X_test_numeric = X_test[num_cols]
            
            for feature in stat_features:
                if feature == 'row_sum':
                    X_test[feature] = X_test_numeric.sum(axis=1)
                elif feature == 'row_mean':
                    X_test[feature] = X_test_numeric.mean(axis=1)
                elif feature == 'row_std':
                    X_test[feature] = X_test_numeric.std(axis=1)
                elif feature == 'row_max':
                    X_test[feature] = X_test_numeric.max(axis=1)
                elif feature == 'row_min':
                    X_test[feature] = X_test_numeric.min(axis=1)
                elif feature == 'row_median':
                    X_test[feature] = X_test_numeric.median(axis=1)
                # Add other stat features as needed
        
        # Apply adaptive preprocessing
        print(f"\n🔧 Applying adaptive preprocessing...")
        current_analysis = self.analyze_data(pd.concat([X, y], axis=1))
        preprocessor = self.smart_preprocessing(X, current_analysis)
        
        X_processed = preprocessor.fit_transform(X)
        X_test_processed = preprocessor.transform(X_test)
        
        # Apply adaptive feature selection
        if apply_selection and X_processed.shape[1] > 50:
            print(f"\n🎯 Applying adaptive feature selection...")
            X_selected, selected_indices, feature_scores = self.adaptive_feature_selection(
                X_processed, y
            )
            X_test_selected = X_test_processed[:, selected_indices]
            
            self.feature_importance_scores = feature_scores
            
            return X_selected, y, X_test_selected, None  # Feature names complex with preprocessing
        
        return X_processed, y, X_test_processed, None


# Integration function for existing pipeline
def build_features_dynamic(train, test):
    """Dynamic feature engineering version"""
    
    engineer = DynamicFeatureEngineer()
    return engineer.build_features_adaptive(train, test)


# Example usage and comparison
if __name__ == "__main__":
    # Demo with sample data
    print("🧪 Dynamic Feature Engineering Demo")
    
    # You would load your actual data here
    # train = pd.read_csv("data/raw/train.csv") 
    # test = pd.read_csv("data/raw/test.csv")
    
    print("This would analyze your data and create features adaptively!")
    print("Features created depend on:")
    print("  • Number of samples (affects preprocessing complexity)")
    print("  • Number of features (affects interaction creation)")
    print("  • Categorical cardinality (affects encoding strategy)")
    print("  • Feature correlations (affects interaction selection)")
    print("  • Missing data patterns (affects imputation strategy)")