import argparse
import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    # Load parameters
    with open(args.params) as f:
        all_params = yaml.safe_load(f)

    prepare_params = all_params["prepare"]
    features_params = all_params["features"]

    print(f"Loading cleaned data from {prepare_params['output_file']}...")
    df = pd.read_csv(prepare_params['output_file'])
    print(f"Data shape: {df.shape}")

    # Separate features and target
    numeric_features = prepare_params['numeric_features']
    categorical_features = prepare_params['categorical_features']
    target = prepare_params['target']

    # Create label encoders for categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} categories")

    # Prepare X and y
    all_features = numeric_features + categorical_features
    X = df[all_features].values
    y = df[target].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=features_params['test_size'],
        random_state=features_params['random_state']
    )

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Save splits
    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)

    # Save label encoders and feature names for Flask app
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(all_features, "models/model_features.pkl")

    # Create feature field map for Flask form
    feature_field_map = {feat: feat.lower().replace(" ", "_") for feat in all_features}
    joblib.dump(feature_field_map, "models/feature_field_map.pkl")

    print("Train/test data and artifacts saved successfully")
