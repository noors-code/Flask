import argparse
import os
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    # Load parameters
    with open(args.params) as f:
        params = yaml.safe_load(f)["evaluate"]

    print("Loading test data...")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    print("Loading model...")
    model = joblib.load("models/house_price_model.pkl")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {}

    if 'r2_score' in params['metrics']:
        metrics['r2_score'] = float(r2_score(y_test, y_pred))
        print(f"R² Score: {metrics['r2_score']:.4f}")

    if 'mean_absolute_error' in params['metrics']:
        metrics['mean_absolute_error'] = float(mean_absolute_error(y_test, y_pred))
        print(f"Mean Absolute Error: {metrics['mean_absolute_error']:,.2f} PKR")

    if 'mean_squared_error' in params['metrics']:
        metrics['mean_squared_error'] = float(mean_squared_error(y_test, y_pred))
        print(f"Mean Squared Error: {metrics['mean_squared_error']:,.2f}")

    if 'root_mean_squared_error' in params['metrics']:
        metrics['root_mean_squared_error'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(f"Root Mean Squared Error: {metrics['root_mean_squared_error']:,.2f} PKR")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/eval.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Show some sample predictions
    print("\nSample Predictions (First 5):")
    print("Actual\t\tPredicted\tDifference")
    for i in range(min(5, len(y_test))):
        diff = y_pred[i] - y_test[i]
        print(f"{y_test[i]:,.0f}\t{y_pred[i]:,.0f}\t{diff:,.0f}")
