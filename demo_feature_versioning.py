#!/usr/bin/env python3
# ===============================
# demo_feature_versioning.py - Feature Versioning Demo
# ===============================
"""
Demonstration of the feature versioning system.

This script shows how to:
1. Create different versions of features
2. Train models with different feature versions
3. Compare performance across versions
4. Manage feature storage efficiently
"""

import pandas as pd
import numpy as np
from pathlib import Path


def demo_feature_creation():
    """Demo: Create different feature versions"""
    print("🎯 DEMO: Creating Feature Versions")
    print("=" * 50)
    
    # Check if we have sample data
    if not Path("data/raw/train.csv").exists():
        print("❌ No training data found. Please add data/raw/train.csv first.")
        return False
    
    print("📋 Available feature versions:")
    print("   v1 - Basic preprocessing (median imputation + one-hot encoding)")
    print("   v2 - Enhanced features (interactions, scaling, max categories)")
    print("   v3 - Advanced features (statistical features + feature selection)")
    
    from feature_manager import create_feature_version
    
    # Create all versions
    versions_to_create = ["v1", "v2", "v3"]
    
    for version in versions_to_create:
        print(f"\n🔨 Creating feature version {version}...")
        success = create_feature_version(version, force=True)
        if success:
            print(f"✅ {version} created successfully!")
        else:
            print(f"❌ Failed to create {version}")
    
    return True


def demo_feature_comparison():
    """Demo: Compare different feature versions"""
    print("\n🎯 DEMO: Feature Version Comparison")
    print("=" * 50)
    
    from feature_manager import compare_feature_versions
    
    # Compare v1 vs v2
    print("📊 Comparing v1 (basic) vs v2 (enhanced):")
    compare_feature_versions("v1", "v2")
    
    print("\n" + "-" * 30)
    
    # Compare v2 vs v3
    print("📊 Comparing v2 (enhanced) vs v3 (advanced):")
    compare_feature_versions("v2", "v3")


def demo_model_training():
    """Demo: Train models with different feature versions"""
    print("\n🎯 DEMO: Training with Different Feature Versions")  
    print("=" * 50)
    
    # Import training function
    from train import run_training
    
    # Train LightGBM with each feature version
    versions = ["v1", "v2", "v3"]
    results = {}
    
    for version in versions:
        print(f"\n🚀 Training LightGBM with features {version}...")
        
        try:
            # Train with minimal Optuna trials for demo (faster)
            score = run_training(
                model_type="lightgbm",
                use_optuna=True,
                n_trials=10,  # Small number for demo
                feature_version=version,
                use_feature_cache=True
            )
            
            results[version] = score
            print(f"✅ {version}: {score:.5f}")
            
        except Exception as e:
            print(f"❌ Failed to train with {version}: {e}")
            results[version] = None
    
    # Show results summary
    print(f"\n{'='*30}")
    print("📈 TRAINING RESULTS SUMMARY")
    print(f"{'='*30}")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
        
        for version, score in sorted_results:
            print(f"  {version}: {score:.5f}")
        
        best_version, best_score = sorted_results[0]
        print(f"\n🏆 Best performing: {best_version} ({best_score:.5f})")
    else:
        print("❌ No successful training runs.")


def demo_feature_benchmark():
    """Demo: Benchmark all feature versions"""
    print("\n🎯 DEMO: Feature Version Benchmarking")
    print("=" * 50)
    
    from feature_manager import benchmark_feature_versions
    
    print("🏁 Running quick benchmark on all feature versions...")
    print("   (Using simple LogisticRegression with 3-fold CV)")
    
    benchmark_feature_versions()


def demo_workflow_examples():
    """Demo: Show practical workflow examples"""
    print("\n🎯 DEMO: Practical Workflows")
    print("=" * 50)
    
    print("💡 Example workflows:")
    print()
    
    print("1️⃣ RAPID EXPERIMENTATION:")
    print("   # Create and test new features quickly")
    print("   python feature_manager.py create v4 --force")
    print("   python train.py lightgbm --features v4 --trials 20")
    print()
    
    print("2️⃣ FEATURE COMPARISON:")
    print("   # Compare two feature versions")
    print("   python feature_manager.py compare v2 v3")
    print("   python feature_manager.py benchmark")
    print()
    
    print("3️⃣ PRODUCTION TRAINING:")
    print("   # Train all models with best features")
    print("   python train.py all --features v3 --trials 100")
    print("   python ensemble_runner.py")
    print()
    
    print("4️⃣ CACHE MANAGEMENT:")
    print("   # List all versions")
    print("   python feature_manager.py list")
    print("   # Clean old versions (keep latest 3)")
    print("   python feature_manager.py clean --keep 3")
    print()
    
    print("5️⃣ ADVANCED FEATURES:")
    print("   # No-cache rebuilding")
    print("   python train.py lightgbm --features v2 --no-cache")
    print("   # Skip hyperparameter optimization")
    print("   python train.py lightgbm --features v1 --no-optuna")


def demo_file_structure():
    """Show the file structure created by feature versioning"""
    print("\n🎯 DEMO: File Structure")
    print("=" * 50)
    
    print("📁 Feature versioning creates this structure:")
    print("data/processed/features/")
    print("├── v1/")
    print("│   ├── X_train.npy")
    print("│   ├── X_test.npy") 
    print("│   ├── y_train.npy")
    print("│   ├── feature_names.csv")
    print("│   └── metadata.csv")
    print("├── v2/")
    print("│   └── ... (same structure)")
    print("└── v3/")
    print("    └── ... (same structure)")
    print()
    
    print("📊 Each version stores:")
    print("  • Processed feature matrices (X_train, X_test)")
    print("  • Target values (y_train)")
    print("  • Feature names (if available)")
    print("  • Metadata (shapes, creation time)")


def main():
    """Run complete feature versioning demonstration"""
    print("🚀 Feature Versioning System Demo")
    print("=" * 60)
    print("This demo shows how to create, manage, and use feature versions")
    print("in your machine learning pipeline.")
    print()
    
    # Check prerequisites
    if not Path("data/raw").exists():
        print("⚠️  Creating data/raw directory...")
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        print("📝 Please add train.csv and test.csv to data/raw/ before running the demo.")
        return
    
    if not Path("data/raw/train.csv").exists():
        print("📝 Please add train.csv to data/raw/ to run the full demo.")
        print("   For now, showing workflow examples...")
        demo_workflow_examples()
        demo_file_structure()
        return
        
    try:
        # Run demo steps
        success = demo_feature_creation()
        if success:
            demo_feature_comparison()
            demo_feature_benchmark()
            demo_workflow_examples()
            demo_file_structure()
            
            # Optionally run model training (commented out for speed)
            print("\n" + "=" * 60)
            print("🎓 DEMO COMPLETE!")
            print("=" * 60)
            print("You can now:")
            print("  • Explore different feature versions with: python feature_manager.py")
            print("  • Train models with versions: python train.py lightgbm --features v2")
            print("  • Run benchmarks: python feature_manager.py benchmark")
            print("  • Compare versions: python feature_manager.py compare v1 v2")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()