"""
Simple test script to verify the new dataset.
"""
import pandas as pd
from src.data.loader import DataLoader

def test_new_dataset():
    """Test that the new dataset loads and validates correctly."""
    print("=" * 60)
    print("Testing New Dataset")
    print("=" * 60)

    try:
        # Initialize data loader
        loader = DataLoader()

        # Load data
        print("\n1. Loading dataset...")
        data = loader.load_data("online_shoppers_intention_new.csv")
        print(f"   ✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

        # Validate data
        print("\n2. Validating dataset...")
        is_valid = loader.validate_data(data)
        if is_valid:
            print("   ✅ Data validation passed")
        else:
            print("   ❌ Data validation failed")
            return False

        # Check basic info
        print("\n3. Dataset information:")
        print(f"   - Total records: {len(data):,}")
        print(f"   - Total features: {len(data.columns)}")
        print(f"   - Missing values: {data.isnull().sum().sum()}")

        # Check target distribution
        print("\n4. Target variable (Revenue) distribution:")
        revenue_counts = data['Revenue'].value_counts()
        for value, count in revenue_counts.items():
            pct = (count / len(data)) * 100
            print(f"   - {value}: {count:,} ({pct:.2f}%)")

        # Show sample data
        print("\n5. Sample data (first 3 rows):")
        print(data.head(3).to_string())

        # Get statistics
        stats = loader.get_data_statistics()
        print("\n6. Dataset statistics:")
        print(f"   - Missing values by column: {len(stats['missing_values']['missing_by_column'])} columns with missing values")

        print("\n" + "=" * 60)
        print("✅ All tests passed! New dataset is ready to use.")
        print("=" * 60)
        print("\nTo use this dataset in your pipeline:")
        print("  - The config.yaml has been updated to point to 'online_shoppers_intention_new.csv'")
        print("  - Run: python src/pipeline.py")
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
