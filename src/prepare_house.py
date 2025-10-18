import argparse
import pandas as pd
import yaml
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    # Load parameters
    with open(args.params) as f:
        params = yaml.safe_load(f)["prepare"]

    print(f"Loading data from {params['input_file']}...")
    df = pd.read_csv(params['input_file'])
    print(f"Original shape: {df.shape}")

    # Get feature columns
    numeric_features = params['numeric_features']
    categorical_features = params['categorical_features']
    target = params['target']
    all_features = numeric_features + categorical_features + [target]

    # Select only needed columns
    df = df[all_features].copy()

    # Remove rows with missing values in critical columns
    print(f"Removing missing values...")
    df = df.dropna(subset=[target] + numeric_features)

    # Fill missing categorical values with 'Unknown'
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')

    # Remove outliers in price (keep prices between 1M and 1B PKR)
    print(f"Removing price outliers...")
    df = df[(df[target] >= 1000000) & (df[target] <= 1000000000)]

    # Remove outliers in numeric features (using IQR method)
    for col in numeric_features:
        if col not in ['latitude', 'longitude']:  # Don't remove outliers from coordinates
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1) & (df[col] <= Q3)]

    print(f"Final shape: {df.shape}")

    # Save cleaned data
    os.makedirs(os.path.dirname(params['output_file']), exist_ok=True)
    df.to_csv(params['output_file'], index=False)
    print(f"Cleaned data saved to {params['output_file']}")
