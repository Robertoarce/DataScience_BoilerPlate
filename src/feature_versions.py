# ===============================
# feature_versions.py - SIMPLIFIED
# ===============================
import numpy as np
import pandas as pd
from pathlib import Path


def save_features(X_train, X_test, y_train=None, feature_names=None, version="v1"):
    """Save processed features with version control"""
    
    # Create version directory
    features_dir = Path("data/processed/features") / version
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(features_dir / "X_train.npy", X_train)
    np.save(features_dir / "X_test.npy", X_test)
    
    if y_train is not None:
        np.save(features_dir / "y_train.npy", y_train)
    
    if feature_names is not None:
        pd.Series(feature_names).to_csv(
            features_dir / "feature_names.csv", 
            index=False, 
            header=False
        )
    
    # Save metadata
    metadata = {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "version": version,
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    pd.Series(metadata).to_csv(features_dir / "metadata.csv", header=False)
    
    print(f"✅ Features saved to {features_dir}")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    

def load_features(version="v1"):
    """Load processed features from disk"""
    
    features_dir = Path("data/processed/features") / version
    
    if not features_exist(version):
        raise FileNotFoundError(f"Features version '{version}' not found")
    
    # Load arrays
    X_train = np.load(features_dir / "X_train.npy")
    X_test = np.load(features_dir / "X_test.npy")
    
    y_train = None
    if (features_dir / "y_train.npy").exists():
        y_train = np.load(features_dir / "y_train.npy")
    
    feature_names = None
    if (features_dir / "feature_names.csv").exists():
        feature_names = pd.read_csv(
            features_dir / "feature_names.csv", 
            header=None
        )[0].tolist()
    
    print(f"✅ Features loaded from {features_dir}")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, feature_names


def features_exist(version="v1"):
    """Check if processed features exist for this version"""
    features_dir = Path("data/processed/features") / version
    return (
        (features_dir / "X_train.npy").exists() and
        (features_dir / "X_test.npy").exists()
    )


def list_versions():
    """List all available feature versions"""
    versions_dir = Path("data/processed/features")
    if not versions_dir.exists():
        return []
    
    versions = [d.name for d in versions_dir.iterdir() if d.is_dir()]
    return sorted(versions)


def get_metadata(version="v1"):
    """Get metadata for specific version"""
    metadata_file = Path("data/processed/features") / version / "metadata.csv"
    if not metadata_file.exists():
        return None
    
    return pd.read_csv(metadata_file, index_col=0, header=None)[1].to_dict()


def show_versions():
    """Show all available versions with metadata"""
    versions = list_versions()
    if not versions:
        print("No feature versions found")
        return
    
    print("📁 Available feature versions:")
    print("-" * 40)
    
    for version in versions:
        metadata = get_metadata(version)
        if metadata:
            print(f"  {version:<8} | {metadata.get('train_shape', 'Unknown')} | {metadata.get('created_at', 'Unknown')[:19]}")
        else:
            print(f"  {version:<8} | No metadata")


# Simple CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            show_versions()
        elif command == "exists":
            version = sys.argv[2] if len(sys.argv) > 2 else "v1"
            exists = features_exist(version)
            print(f"Version '{version}' exists: {exists}")
        else:
            print("Usage: python feature_versions.py [list|exists]")
    else:
        show_versions()