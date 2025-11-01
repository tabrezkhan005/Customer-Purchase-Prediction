"""
Script to generate a synthetic dataset for customer purchase prediction.
This dataset maintains similar statistical properties to the original dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import argparse


class SyntheticDatasetGenerator:
    """Generate synthetic online shoppers intention dataset."""

    def __init__(self, n_samples: int = 12330, random_seed: int = 42):
        """
        Initialize the dataset generator.

        Args:
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        np.random.seed(random_seed)

    def generate_numeric_features(self) -> Dict[str, np.ndarray]:
        """Generate numeric features based on original distributions."""
        data = {}

        # Administrative (integer, 0-27, right-skewed)
        data['Administrative'] = np.random.negative_binomial(2, 0.3, self.n_samples)
        data['Administrative'] = np.clip(data['Administrative'], 0, 27)

        # Administrative_Duration (float, correlated with Administrative)
        base_duration = data['Administrative'] * 35
        noise = np.random.normal(0, 20, self.n_samples)
        data['Administrative_Duration'] = np.maximum(0, base_duration + noise)

        # Informational (integer, 0-10, zero-inflated)
        data['Informational'] = np.random.choice([0, 1, 2, 3], size=self.n_samples,
                                                  p=[0.85, 0.10, 0.04, 0.01])

        # Informational_Duration (float, correlated with Informational)
        base_duration = data['Informational'] * 60
        noise = np.random.normal(0, 30, self.n_samples)
        data['Informational_Duration'] = np.maximum(0, base_duration + noise)

        # ProductRelated (integer, 1-100+, right-skewed)
        data['ProductRelated'] = np.random.negative_binomial(3, 0.15, self.n_samples) + 1
        data['ProductRelated'] = np.clip(data['ProductRelated'], 1, 200)

        # ProductRelated_Duration (float, correlated with ProductRelated)
        base_duration = data['ProductRelated'] * 75
        noise = np.random.normal(0, 100, self.n_samples)
        data['ProductRelated_Duration'] = np.maximum(0, base_duration + noise)

        # BounceRates (float, 0-1, mostly low values)
        data['BounceRates'] = np.random.beta(2, 10, self.n_samples)

        # ExitRates (float, 0-1, slightly higher than bounce rates)
        data['ExitRates'] = np.random.beta(2, 8, self.n_samples)

        # PageValues (float, 0-200+, heavily zero-inflated, right-skewed)
        has_value = np.random.choice([0, 1], size=self.n_samples, p=[0.90, 0.10])
        values = np.random.gamma(2, 10, self.n_samples)
        data['PageValues'] = has_value * values

        # SpecialDay (float, 0-1, mostly zeros)
        specialday_probs = [0.70, 0.10, 0.08, 0.06, 0.04, 0.02]
        specialday_probs = np.array(specialday_probs) / np.sum(specialday_probs)  # Normalize
        data['SpecialDay'] = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                             size=self.n_samples,
                                             p=specialday_probs)

        # OperatingSystems (integer, 1-8, concentrated around 1-4)
        os_probs = [0.50, 0.25, 0.15, 0.05, 0.02, 0.02, 0.005, 0.005]
        os_probs = np.array(os_probs) / np.sum(os_probs)  # Normalize
        data['OperatingSystems'] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],
                                                   size=self.n_samples,
                                                   p=os_probs)

        # Browser (integer, 1-13, concentrated around 1-6)
        browser_probs = [0.30, 0.25, 0.20, 0.10, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
        browser_probs = np.array(browser_probs) / np.sum(browser_probs)  # Normalize
        data['Browser'] = np.random.choice(range(1, 14),
                                          size=self.n_samples,
                                          p=browser_probs)

        # Region (integer, 1-9, concentrated around 1-4)
        region_probs = [0.30, 0.25, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]
        region_probs = np.array(region_probs) / np.sum(region_probs)  # Normalize
        data['Region'] = np.random.choice(range(1, 10),
                                         size=self.n_samples,
                                         p=region_probs)

        # TrafficType (integer, 1-20, concentrated around 1-5)
        traffic_weights = [0.30, 0.25, 0.20, 0.10, 0.05] + [0.02] * 15
        traffic_weights = np.array(traffic_weights) / np.sum(traffic_weights)  # Normalize
        data['TrafficType'] = np.random.choice(range(1, 21), size=self.n_samples, p=traffic_weights)

        return data

    def generate_categorical_features(self) -> Dict[str, np.ndarray]:
        """Generate categorical features."""
        data = {}

        # Month (10 months from original)
        month_probs = [0.10, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.10, 0.12, 0.08]
        month_probs = np.array(month_probs) / np.sum(month_probs)  # Normalize
        data['Month'] = np.random.choice(['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                        size=self.n_samples,
                                        p=month_probs)

        # VisitorType
        visitortype_probs = [0.85, 0.13, 0.02]
        visitortype_probs = np.array(visitortype_probs) / np.sum(visitortype_probs)  # Normalize
        data['VisitorType'] = np.random.choice(['Returning_Visitor', 'New_Visitor', 'Other'],
                                              size=self.n_samples,
                                              p=visitortype_probs)

        # Weekend (boolean)
        data['Weekend'] = np.random.choice([True, False], size=self.n_samples, p=[0.25, 0.75])

        return data

    def generate_target(self, numeric_data: Dict[str, np.ndarray],
                       categorical_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate target variable (Revenue) based on feature relationships.
        Higher PageValues, more ProductRelated pages, and Returning_Visitor
        should correlate with higher purchase probability.
        """
        # Base probability
        base_prob = 0.15

        # Feature contributions
        page_value_contrib = np.clip(numeric_data['PageValues'] / 100, 0, 0.3)
        product_contrib = np.clip(numeric_data['ProductRelated'] / 50, 0, 0.2)
        bounce_penalty = numeric_data['BounceRates'] * 0.15
        returning_bonus = (categorical_data['VisitorType'] == 'Returning_Visitor').astype(float) * 0.1
        weekend_bonus = categorical_data['Weekend'].astype(float) * 0.05

        # Calculate probability
        prob = (base_prob +
                page_value_contrib +
                product_contrib -
                bounce_penalty +
                returning_bonus +
                weekend_bonus +
                np.random.normal(0, 0.05, self.n_samples))

        # Clip to valid probability range
        prob = np.clip(prob, 0, 1)

        # Generate binary target
        target = np.random.binomial(1, prob, self.n_samples).astype(bool)

        # Ensure reasonable class balance (around 15% positive)
        positive_count = np.sum(target)
        if positive_count < self.n_samples * 0.12:
            # Add more positives
            idx_to_flip = np.random.choice(np.where(~target)[0],
                                         size=int(self.n_samples * 0.15 - positive_count),
                                         replace=False)
            target[idx_to_flip] = True
        elif positive_count > self.n_samples * 0.18:
            # Remove some positives
            idx_to_flip = np.random.choice(np.where(target)[0],
                                         size=int(positive_count - self.n_samples * 0.15),
                                         replace=False)
            target[idx_to_flip] = False

        return target

    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete synthetic dataset."""
        print(f"Generating synthetic dataset with {self.n_samples} samples...")

        # Generate features
        numeric_data = self.generate_numeric_features()
        categorical_data = self.generate_categorical_features()

        # Generate target
        target = self.generate_target(numeric_data, categorical_data)

        # Combine all data
        all_data = {**numeric_data, **categorical_data, 'Revenue': target}

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Ensure correct column order
        expected_order = [
            'Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
            'OperatingSystems', 'Browser', 'Region', 'TrafficType',
            'VisitorType', 'Weekend', 'Revenue'
        ]
        df = df[expected_order]

        # Convert boolean columns properly
        df['Weekend'] = df['Weekend'].astype(bool)
        df['Revenue'] = df['Revenue'].astype(bool)

        # Ensure integer columns are integers
        int_columns = ['Administrative', 'Informational', 'ProductRelated',
                      'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        for col in int_columns:
            df[col] = df[col].astype(int)

        print(f"Dataset generated successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nRevenue distribution:")
        print(df['Revenue'].value_counts())
        print(f"Purchase rate: {df['Revenue'].mean()*100:.2f}%")

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str = "online_shoppers_intention_synthetic.csv"):
        """Save dataset to CSV file."""
        # Convert boolean columns to string format matching original (TRUE/FALSE)
        df_export = df.copy()
        df_export['Weekend'] = df_export['Weekend'].map({True: 'TRUE', False: 'FALSE'})
        df_export['Revenue'] = df_export['Revenue'].map({True: 'TRUE', False: 'FALSE'})

        df_export.to_csv(filepath, index=False)
        print(f"\nDataset saved to: {filepath}")


def main():
    """Main function to generate synthetic dataset."""
    parser = argparse.ArgumentParser(description='Generate synthetic online shoppers intention dataset')
    parser.add_argument('--n_samples', type=int, default=12330,
                       help='Number of samples to generate (default: 12330)')
    parser.add_argument('--output', type=str, default='online_shoppers_intention_synthetic.csv',
                       help='Output filename (default: online_shoppers_intention_synthetic.csv)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Generate dataset
    generator = SyntheticDatasetGenerator(n_samples=args.n_samples, random_seed=args.seed)
    df = generator.generate_dataset()

    # Save dataset
    generator.save_dataset(df, args.output)

    print(f"\n✅ Synthetic dataset generation complete!")
    print(f"File: {args.output}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
