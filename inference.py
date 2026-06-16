"""
inference.py — Customer Churn Prediction
=========================================
Standalone inference script for the customer churn prediction model.

Trains a Random Forest classifier on the full synthetic dataset and exposes:
  - predict_churn(customer: dict) -> dict
  - batch_predict(customers: list[dict]) -> list[dict]

Usage (single prediction):
    python inference.py

Usage (as a module):
    from inference import predict_churn
    result = predict_churn({"Age": 35, "Balance": 75000, ...})
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["Age", "Balance", "Salary", "Tenure", "NumProducts", "IsActive"]
RANDOM_STATE = 42
N_SAMPLES = 1000

# ---------------------------------------------------------------------------
# Data generation (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """Generate the synthetic customer churn dataset."""
    np.random.seed(RANDOM_STATE)
    data = pd.DataFrame({
        "Age":         np.random.randint(18, 60, n),
        "Balance":     np.random.randint(1000, 100000, n),
        "Salary":      np.random.randint(20000, 150000, n),
        "Tenure":      np.random.randint(1, 10, n),
        "NumProducts": np.random.randint(1, 4, n),
        "IsActive":    np.random.randint(0, 2, n),
    })
    # Deterministic churn rule from the notebook
    data["Churn"] = ((data["Balance"] > 50000) & (data["IsActive"] == 0)).astype(int)
    return data


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(data: pd.DataFrame):
    """
    Train a Random Forest classifier and fit a StandardScaler.

    Returns:
        model    (RandomForestClassifier) — trained model
        scaler   (StandardScaler)         — fitted scaler
        metrics  (dict)                   — test-set performance summary
    """
    X = data[FEATURE_COLUMNS]
    y = data["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# Global model — loaded once at import time
# ---------------------------------------------------------------------------

_data   = generate_dataset()
_model, _scaler, _train_metrics = train_model(_data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_churn(customer: dict) -> dict:
    """
    Predict churn for a single customer.

    Args:
        customer (dict): Dictionary with keys:
            Age          (int)  — 18 to 59
            Balance      (int)  — 1000 to 100000
            Salary       (int)  — 20000 to 150000
            Tenure       (int)  — 1 to 9
            NumProducts  (int)  — 1 to 3
            IsActive     (int)  — 0 or 1

    Returns:
        dict:
            prediction        (int)   — 1 = Churn, 0 = No Churn
            churn_probability (float) — probability of churn (0.0 – 1.0)
            label             (str)   — "Churn" or "No Churn"

    Raises:
        ValueError: If any required feature is missing.

    Example:
        >>> result = predict_churn({
        ...     "Age": 45, "Balance": 80000, "Salary": 55000,
        ...     "Tenure": 2, "NumProducts": 1, "IsActive": 0
        ... })
        >>> print(result)
        {'prediction': 1, 'churn_probability': 0.92, 'label': 'Churn'}
    """
    missing = [col for col in FEATURE_COLUMNS if col not in customer]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Build input array in the correct column order
    input_df = pd.DataFrame([{col: customer[col] for col in FEATURE_COLUMNS}])
    input_scaled = _scaler.transform(input_df)

    prediction = int(_model.predict(input_scaled)[0])
    probability = round(float(_model.predict_proba(input_scaled)[0][1]), 4)

    return {
        "prediction":        prediction,
        "churn_probability": probability,
        "label":             "Churn" if prediction == 1 else "No Churn",
    }


def batch_predict(customers: list) -> list:
    """
    Predict churn for a list of customers.

    Args:
        customers (list[dict]): List of customer dictionaries.
                                Each dict must contain all FEATURE_COLUMNS keys.

    Returns:
        list[dict]: List of prediction result dicts (same schema as predict_churn).

    Example:
        >>> results = batch_predict([
        ...     {"Age": 25, "Balance": 30000, "Salary": 40000,
        ...      "Tenure": 5, "NumProducts": 2, "IsActive": 1},
        ...     {"Age": 52, "Balance": 90000, "Salary": 75000,
        ...      "Tenure": 1, "NumProducts": 1, "IsActive": 0},
        ... ])
    """
    if not customers:
        return []

    missing_per_row = [
        [col for col in FEATURE_COLUMNS if col not in c]
        for c in customers
    ]
    errors = [(i, m) for i, m in enumerate(missing_per_row) if m]
    if errors:
        raise ValueError(f"Missing features in rows: {errors}")

    input_df = pd.DataFrame(
        [{col: c[col] for col in FEATURE_COLUMNS} for c in customers]
    )
    input_scaled = _scaler.transform(input_df)

    predictions  = _model.predict(input_scaled).tolist()
    probabilities = _model.predict_proba(input_scaled)[:, 1].tolist()

    return [
        {
            "prediction":        int(pred),
            "churn_probability": round(float(prob), 4),
            "label":             "Churn" if pred == 1 else "No Churn",
        }
        for pred, prob in zip(predictions, probabilities)
    ]


def get_model_info() -> dict:
    """
    Return metadata about the loaded model and its training performance.

    Returns:
        dict:
            model_type    (str)  — classifier class name
            n_estimators  (int)  — number of trees in the forest
            features      (list) — ordered list of feature names
            train_metrics (dict) — accuracy, f1_score, roc_auc on test split
    """
    return {
        "model_type":    type(_model).__name__,
        "n_estimators":  _model.n_estimators,
        "features":      FEATURE_COLUMNS,
        "train_metrics": _train_metrics,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Customer Churn Prediction — Inference Demo")
    print("=" * 55)

    # Model info
    info = get_model_info()
    print(f"\nModel   : {info['model_type']} ({info['n_estimators']} trees)")
    print(f"Features: {info['features']}")
    print(f"Test Accuracy : {info['train_metrics']['accuracy']}")
    print(f"Test F1 Score : {info['train_metrics']['f1_score']}")
    print(f"Test ROC-AUC  : {info['train_metrics']['roc_auc']}")

    # --- Single prediction examples ---
    print("\n--- Single Prediction Examples ---")

    high_risk = {
        "Age": 45, "Balance": 85000, "Salary": 55000,
        "Tenure": 2, "NumProducts": 1, "IsActive": 0
    }
    low_risk = {
        "Age": 28, "Balance": 20000, "Salary": 42000,
        "Tenure": 6, "NumProducts": 3, "IsActive": 1
    }

    for label, customer in [("High-Risk Customer", high_risk), ("Low-Risk Customer", low_risk)]:
        result = predict_churn(customer)
        print(f"\n{label}:")
        print(f"  Input  : {customer}")
        print(f"  Result : {result}")

    # --- Batch prediction example ---
    print("\n--- Batch Prediction (5 customers) ---")
    batch = [
        {"Age": 50, "Balance": 92000, "Salary": 70000, "Tenure": 1, "NumProducts": 1, "IsActive": 0},
        {"Age": 30, "Balance": 15000, "Salary": 35000, "Tenure": 7, "NumProducts": 2, "IsActive": 1},
        {"Age": 40, "Balance": 60000, "Salary": 80000, "Tenure": 4, "NumProducts": 2, "IsActive": 0},
        {"Age": 23, "Balance": 5000,  "Salary": 22000, "Tenure": 1, "NumProducts": 1, "IsActive": 1},
        {"Age": 55, "Balance": 75000, "Salary": 95000, "Tenure": 3, "NumProducts": 1, "IsActive": 0},
    ]
    batch_results = batch_predict(batch)
    for i, (customer, result) in enumerate(zip(batch, batch_results), 1):
        print(f"  Customer {i}: Balance={customer['Balance']}, IsActive={customer['IsActive']} "
              f"→ {result['label']} (p={result['churn_probability']})")

    print("\nDone.")
