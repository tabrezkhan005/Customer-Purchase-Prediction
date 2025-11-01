"""
Quick test script to verify the new dataset works with the pipeline.
"""
from src.pipeline import MLPipeline
import pandas as pd

def test_new_dataset():
    """Test that the new dataset loads and validates correctly."""
    print("=" * 60)
    print("Testing New Dataset with ML Pipeline")
    print("=" * 60)

    try:
        # Initialize pipeline with config (should use new dataset)
        pipeline = MLPipeline(config_path="config.yaml")

        # Load data
        print("\n1. Loading dataset...")
        data = pipeline.load_data()
        print(f"   ✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

        # Check basic info
        print("\n2. Dataset information:")
        print(f"   - Total records: {len(data):,}")
        print(f"   - Total features: {len(data.columns)}")
        print(f"   - Missing values: {data.isnull().sum().sum()}")

        # Check target distribution
        print("\n3. Target variable (Revenue) distribution:")
        revenue_counts = data['Revenue'].value_counts()
        for value, count in revenue_counts.items():
            pct = (count / len(data)) * 100
            print(f"   - {value}: {count:,} ({pct:.2f}%)")

        # Test preprocessing
        print("\n4. Testing preprocessing...")
        X, y, feature_names = pipeline.preprocess_data(data)
        print(f"   ✅ Preprocessing successful")
        print(f"   - Features shape: {X.shape}")
        print(f"   - Target shape: {y.shape}")
        print(f"   - Number of features: {len(feature_names)}")

        print("\n" + "=" * 60)
        print("✅ All tests passed! New dataset is ready to use.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_new_dataset()
    exit(0 if success else 1)
