# ===============================
# titanic_example.py - Dynamic Features with Titanic Dataset
# ===============================
import pandas as pd
import numpy as np
from dynamic_features import DynamicFeatureEngineer

def create_sample_titanic_data():
    """Create sample Titanic-like data to demonstrate dynamic feature engineering"""
    
    # Simulate Titanic dataset structure
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        # Numerical features
        'Age': np.random.normal(30, 12, n_samples),
        'Fare': np.random.exponential(20, n_samples),
        'SibSp': np.random.poisson(0.5, n_samples),  # Siblings/Spouses
        'Parch': np.random.poisson(0.3, n_samples),  # Parents/Children
        
        # Categorical features  
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.2, 0.6]),
        
        # Target (Survived: 1=Yes, 0=No)
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    # Add some realistic correlations
    df = pd.DataFrame(data)
    
    # Make survival more likely for females, higher class, younger people
    survival_prob = 0.3  # Base probability
    survival_prob += (df['Sex'] == 'female') * 0.4  # Females more likely
    survival_prob += (df['Pclass'] == 1) * 0.3      # First class more likely
    survival_prob += (df['Age'] < 18) * 0.2         # Children more likely
    survival_prob -= (df['Age'] > 60) * 0.1         # Elderly less likely
    
    # Generate realistic survival target
    df['target'] = np.random.binomial(1, np.clip(survival_prob, 0, 1), n_samples)
    
    return df

def demonstrate_dynamic_analysis():
    """Show what the dynamic system analyzes about Titanic data"""
    
    print("🚢 TITANIC DATASET EXAMPLE")
    print("=" * 50)
    
    # Create sample data
    train_data = create_sample_titanic_data()
    test_data = create_sample_titanic_data()
    
    print("📋 Original Titanic data structure:")
    print(f"   Samples: {len(train_data)}")
    print(f"   Features: {train_data.drop('target', axis=1).shape[1]}")
    print(f"   Columns: {list(train_data.drop('target', axis=1).columns)}")
    print()
    
    # Initialize dynamic feature engineer
    engineer = DynamicFeatureEngineer(target_col='target')
    
    # STEP 1: Automatic data analysis
    print("🔍 STEP 1: Automatic Data Analysis")
    print("-" * 30)
    analysis = engineer.analyze_data(train_data)
    
    print(f"📊 Dataset characteristics (detected automatically):")
    print(f"   • Samples: {analysis['n_samples']}")
    print(f"   • Original features: {analysis['n_features']}")
    print(f"   • Numerical: {len(analysis['numerical_cols'])} → {analysis['numerical_cols']}")
    print(f"   • Categorical: {len(analysis['categorical_cols'])} → {analysis['categorical_cols']}")
    print(f"   • High cardinality cats: {analysis['high_cardinality_cats']}")
    
    # Show target correlations
    print(f"\n🎯 Target correlations (automatic):")
    for feature, corr in analysis['target_correlation'].items():
        print(f"   • {feature}: {corr:.3f}")
    
    return train_data, test_data, engineer, analysis

def show_automatic_feature_creation():
    """Demonstrate automatic feature creation process"""
    
    print("\n" + "=" * 50)
    print("🔧 STEP 2: Automatic Feature Creation")
    print("=" * 50)
    
    train_data, test_data, engineer, analysis = demonstrate_dynamic_analysis()
    
    # Show what features will be created automatically
    print("\n🧠 Dynamic system decides automatically:")
    print("   ✅ Create interactions? → YES (4 numerical features found)")
    print("   ✅ Create statistics? → YES (4 numerical features >= 3 minimum)")  
    print("   ✅ Apply feature selection? → YES (will create many features)")
    print()
    
    # STEP 2: Automatic interaction creation
    print("🔗 AUTOMATIC INTERACTION CREATION:")
    print("   • Finds most correlated features: Age, Fare, SibSp, Parch")
    print("   • Creates ONLY useful interactions (not all combinations)")
    print("   • Examples that WILL be created:")
    print("     - Age_x_Fare (age * fare interaction)")
    print("     - Age_div_Fare (age/fare ratio)")  
    print("     - SibSp_x_Parch (family size interaction)")
    print("   • Examples that WON'T be created:")
    print("     - Low correlation features ignored")
    print("     - Redundant combinations skipped")
    
    # STEP 3: Automatic statistical features  
    print("\n📊 AUTOMATIC STATISTICAL FEATURES:")
    print("   • Creates row-wise statistics across numerical features:")
    print("     - row_sum: Age + Fare + SibSp + Parch")
    print("     - row_mean: Average of numerical features")
    print("     - row_std: Standard deviation across features")
    print("     - row_max, row_min, row_median")
    
    # STEP 4: Automatic preprocessing choices
    print("\n🔧 AUTOMATIC PREPROCESSING CHOICES:")
    print("   • Dataset size: 1000 samples → Medium size processing")
    print("   • Numerical: Median imputation + StandardScaler")
    print("   • Categorical encoding decisions:")
    print("     - Sex (2 categories): One-hot encoding")
    print("     - Embarked (3 categories): One-hot encoding") 
    print("     - Pclass (3 categories): One-hot encoding")
    print("   • Feature selection: ~200 features (based on data size)")
    
    return train_data, test_data, engineer

def run_dynamic_features_live():
    """Actually run the dynamic feature engineering"""
    
    print("\n" + "=" * 50)
    print("🚀 STEP 3: Running Dynamic Feature Engineering")
    print("=" * 50)
    
    train_data, test_data, engineer = show_automatic_feature_creation()
    
    # Run the actual dynamic feature engineering
    X_processed, y, X_test_processed, feature_names = engineer.build_features_adaptive(
        train_data, test_data,
        create_interactions=True,
        create_stats=True,
        apply_selection=True
    )
    
    print(f"\n✅ RESULTS:")
    print(f"   • Original features: {train_data.drop('target', axis=1).shape[1]}")
    print(f"   • Final features: {X_processed.shape[1]}")
    print(f"   • Features created automatically: {X_processed.shape[1] - train_data.drop('target', axis=1).shape[1]}")
    print(f"   • Training shape: {X_processed.shape}")
    print(f"   • Test shape: {X_test_processed.shape}")
    
    return X_processed, y, X_test_processed

def compare_with_manual_features():
    """Show what you might add manually vs what's automatic"""
    
    print("\n" + "=" * 50)
    print("🧠 MANUAL vs AUTOMATIC FEATURES")
    print("=" * 50)
    
    print("🤖 AUTOMATIC (Dynamic System Creates):")
    print("   ✅ Age_x_Fare, Age_div_Fare (interaction features)")
    print("   ✅ SibSp_x_Parch (family relationships)")  
    print("   ✅ row_sum, row_mean, row_std (statistical features)")
    print("   ✅ Optimal preprocessing (scaling, encoding)")
    print("   ✅ Feature selection (keeps best ~200 features)")
    print()
    
    print("👤 MANUAL (You might still want to add):")
    print("   🔧 Domain-specific features:")
    print("      - FamilySize = SibSp + Parch + 1")
    print("      - IsAlone = (FamilySize == 1)")
    print("      - Title = extract from Name ('Mr', 'Mrs', 'Miss')")
    print("      - AgeGroup = binned age categories")
    print("      - FarePerPerson = Fare / FamilySize")
    print()
    
    print("💡 RECOMMENDED APPROACH:")
    print("   1. Start with dynamic features (automatic)")
    print("   2. Add domain knowledge features (manual)")
    print("   3. Let system choose best combination")

def create_enhanced_version():
    """Show how to combine automatic + manual features"""
    
    print("\n" + "=" * 50) 
    print("🎯 ENHANCED VERSION: Automatic + Manual")
    print("=" * 50)
    
    print("📝 Code example for Titanic with both approaches:")
    print()
    
    code_example = '''
def build_features_titanic_enhanced(train, test):
    """Titanic: Automatic dynamic features + manual domain features"""
    
    # STEP 1: Manual domain-specific features
    def add_titanic_features(df):
        df = df.copy()
        
        # Family features (domain knowledge)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Age groups (domain knowledge) 
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Elder'])
        
        # Fare per person (domain knowledge)
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        
        return df
    
    # Add manual features first
    train_enhanced = add_titanic_features(train)
    test_enhanced = add_titanic_features(test)
    
    # STEP 2: Apply dynamic feature engineering
    engineer = DynamicFeatureEngineer(target_col='target')
    X, y, X_test, feature_names = engineer.build_features_adaptive(
        train_enhanced, test_enhanced,
        create_interactions=True,  # Automatic interactions
        create_stats=True,         # Automatic statistics  
        apply_selection=True       # Automatic feature selection
    )
    
    return X, y, X_test, feature_names
    '''
    
    print(code_example)
    print("\n🎯 This approach gives you:")
    print("   • 🤖 Automatic: Smart interactions, statistics, preprocessing")
    print("   • 🧠 Manual: Domain expertise (family patterns, age groups)")
    print("   • 🎯 Optimal: System selects best combination automatically")

def main():
    """Full demonstration of dynamic feature engineering"""
    
    print("🎯 DYNAMIC FEATURE ENGINEERING EXPLAINED")
    print("Using Titanic Dataset as Example")
    print("=" * 60)
    
    # Run complete demonstration
    demonstrate_dynamic_analysis()
    show_automatic_feature_creation()
    run_dynamic_features_live()
    compare_with_manual_features()
    create_enhanced_version()
    
    print("\n" + "=" * 60)
    print("🎓 SUMMARY: How Dynamic Features Work")
    print("=" * 60)
    print("✅ AUTOMATIC: System analyzes your data and creates features")
    print("✅ ADAPTIVE: Different strategies for different datasets")  
    print("✅ SMART: Only creates useful features, not everything")
    print("✅ FLEXIBLE: You can still add domain-specific features")
    print("✅ CACHED: Works with your versioning system")
    print()
    print("🚀 Usage:")
    print("   python src/train.py lightgbm --features v4  # Dynamic features")
    print("   python src/feature_manager.py compare v2 v4  # Compare approaches")

if __name__ == "__main__":
    main()