# ===============================
# feature_manager.py - Feature Management Utility
# ===============================
import pandas as pd
import numpy as np
from pathlib import Path
from features import build_features
from feature_versions import show_versions, get_metadata, features_exist
import argparse


def create_feature_version(version, force=False):
    """Create a new feature version"""
    
    if features_exist(version) and not force:
        print(f"❌ Feature version '{version}' already exists. Use --force to overwrite.")
        return False
        
    try:
        # Load raw data
        print(f"📥 Loading raw data...")
        train = pd.read_csv("data/raw/train.csv")
        test = pd.read_csv("data/raw/test.csv")
        
        # Build features (this will automatically save them)
        print(f"🔨 Building features version {version}...")
        X, y, X_test, feature_names = build_features(train, test, version=version, use_cache=False)
        
        print(f"✅ Feature version '{version}' created successfully!")
        print(f"   Train shape: {X.shape}")
        print(f"   Test shape: {X_test.shape}")
        if feature_names:
            print(f"   Features: {len(feature_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create feature version '{version}': {e}")
        return False


def compare_feature_versions(version1="v1", version2="v2"):
    """Compare two feature versions"""
    
    print(f"📊 Comparing feature versions {version1} vs {version2}")
    print("-" * 50)
    
    # Check if both versions exist
    if not features_exist(version1):
        print(f"❌ Feature version '{version1}' doesn't exist")
        return
    if not features_exist(version2):
        print(f"❌ Feature version '{version2}' doesn't exist")
        return
    
    # Get metadata
    meta1 = get_metadata(version1)
    meta2 = get_metadata(version2)
    
    if meta1:
        print(f"{version1:>8} | Shape: {meta1.get('train_shape', 'Unknown')} | Created: {meta1.get('created_at', 'Unknown')[:19]}")
    if meta2:
        print(f"{version2:>8} | Shape: {meta2.get('train_shape', 'Unknown')} | Created: {meta2.get('created_at', 'Unknown')[:19]}")
    
    # Load both versions for comparison
    from feature_versions import load_features
    
    try:
        X1, _, _, names1 = load_features(version1)
        X2, _, _, names2 = load_features(version2)
        
        print(f"\nFeature count comparison:")
        print(f"  {version1}: {X1.shape[1]} features")
        print(f"  {version2}: {X2.shape[1]} features")
        print(f"  Difference: {X2.shape[1] - X1.shape[1]:+d} features")
        
        if names1 and names2:
            # Show feature name differences
            set1 = set(names1)
            set2 = set(names2)
            
            only_in_v1 = set1 - set2
            only_in_v2 = set2 - set1
            common = set1.intersection(set2)
            
            print(f"\nFeature overlap:")
            print(f"  Common features: {len(common)}")
            print(f"  Only in {version1}: {len(only_in_v1)}")
            print(f"  Only in {version2}: {len(only_in_v2)}")
            
            if only_in_v2 and len(only_in_v2) <= 10:
                print(f"\n  New in {version2}: {list(only_in_v2)}")
        
    except Exception as e:
        print(f"❌ Failed to load features for comparison: {e}")


def benchmark_feature_versions():
    """Benchmark all available feature versions"""
    
    print("🏁 Feature Version Benchmark")
    print("=" * 40)
    
    from feature_versions import list_versions
    versions = list_versions()
    
    if not versions:
        print("❌ No feature versions found. Create some first!")
        return
    
    # Quick benchmark - train a simple model on each version
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    results = {}
    
    for version in versions:
        try:
            print(f"\n🔍 Benchmarking {version}...")
            
            # Load features
            from feature_versions import load_features
            X, _, y, _ = load_features(version)
            
            # Simple pipeline with logistic regression
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            # 3-fold CV score
            scores = cross_val_score(pipeline, X, y, cv=3, scoring='roc_auc')
            mean_score = scores.mean()
            
            results[version] = mean_score
            print(f"   CV Score: {mean_score:.4f} (±{scores.std():.4f})")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[version] = None
    
    # Summary
    print("\n" + "=" * 40)
    print("📈 BENCHMARK RESULTS")
    print("=" * 40)
    
    sorted_results = sorted([(k, v) for k, v in results.items() if v is not None], 
                           key=lambda x: x[1], reverse=True)
    
    for version, score in sorted_results:
        print(f"{version:>8} | {score:.4f}")
    
    if sorted_results:
        best_version, best_score = sorted_results[0]
        print(f"\n🏆 Best: {best_version} ({best_score:.4f})")


def clean_feature_versions(keep_latest=3):
    """Clean old feature versions, keeping only the latest N"""
    
    from feature_versions import list_versions
    versions = list_versions()
    
    if len(versions) <= keep_latest:
        print(f"✅ Only {len(versions)} versions found, nothing to clean.")
        return
    
    # Sort by version name (assumes v1, v2, v3... naming)
    versions_sorted = sorted(versions)
    to_remove = versions_sorted[:-keep_latest]
    
    print(f"🧹 Cleaning feature versions...")
    print(f"   Keeping latest {keep_latest}: {versions_sorted[-keep_latest:]}")
    print(f"   Removing: {to_remove}")
    
    confirm = input("Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled.")
        return
    
    # Remove old versions
    import shutil
    for version in to_remove:
        version_dir = Path("data/processed/features") / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            print(f"   🗑️ Removed {version}")


def main():
    """CLI interface for feature management"""
    
    parser = argparse.ArgumentParser(description="Feature Version Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List versions
    list_parser = subparsers.add_parser('list', help='List all feature versions')
    
    # Create version
    create_parser = subparsers.add_parser('create', help='Create new feature version')
    create_parser.add_argument('version', help='Version name (e.g., v1, v2, v3)')
    create_parser.add_argument('--force', action='store_true', help='Overwrite if exists')
    
    # Compare versions
    compare_parser = subparsers.add_parser('compare', help='Compare two feature versions')
    compare_parser.add_argument('version1', help='First version to compare')
    compare_parser.add_argument('version2', help='Second version to compare')
    
    # Benchmark versions
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark all versions')
    
    # Clean versions
    clean_parser = subparsers.add_parser('clean', help='Clean old feature versions')
    clean_parser.add_argument('--keep', type=int, default=3, help='Number of versions to keep')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        show_versions()
    elif args.command == 'create':
        create_feature_version(args.version, args.force)
    elif args.command == 'compare':
        compare_feature_versions(args.version1, args.version2)
    elif args.command == 'benchmark':
        benchmark_feature_versions()
    elif args.command == 'clean':
        clean_feature_versions(args.keep)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()