import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    # Load parameters
    with open(args.params) as f:
        params = yaml.safe_load(f)["train"]

    print("Loading training data...")
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Create model
    print(f"Training {params['model_type']} with parameters:")
    for key, value in params.items():
        if key != 'model_type':
            print(f"  {key}: {value}")

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
        n_jobs=params['n_jobs'],
        verbose=1
    )

    # Train model
    print("Training model...")
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/house_price_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Print feature importances
    feature_names = joblib.load("models/model_features.pkl")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Feature Importances:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
