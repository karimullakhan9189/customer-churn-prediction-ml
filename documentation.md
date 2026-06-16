# Technical Documentation — Customer Churn Prediction

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Dataset Generation](#2-dataset-generation)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Model Training](#4-model-training)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Visualizations](#6-visualizations)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Results Interpretation](#8-results-interpretation)
9. [Extending the Project](#9-extending-the-project)
10. [Dependencies Reference](#10-dependencies-reference)

---

## 1. Project Summary

This project implements a multi-model customer churn classification system. It uses a synthetically generated banking dataset to simulate real-world churn prediction scenarios. The pipeline covers data generation, preprocessing, model training, cross-validation, metric evaluation, and visualization — all in a single reproducible notebook with a companion inference script.

**Random seed:** `42` (set globally via `np.random.seed(42)` for full reproducibility)

---

## 2. Dataset Generation

The dataset is programmatically generated using NumPy random functions. No external files or downloads are required.

```python
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'Age':         np.random.randint(18, 60, n),
    'Balance':     np.random.randint(1000, 100000, n),
    'Salary':      np.random.randint(20000, 150000, n),
    'Tenure':      np.random.randint(1, 10, n),
    'NumProducts': np.random.randint(1, 4, n),
    'IsActive':    np.random.randint(0, 2, n)
})
```

### Churn Label Logic

The binary target variable `Churn` is derived from a deterministic rule:

```python
data['Churn'] = ((data['Balance'] > 50000) & (data['IsActive'] == 0)).astype(int)
```

This produces a **realistic class imbalance**: only customers who are both high-balance AND inactive are labeled as churned (~25% of records, depending on random seed).

### Feature Descriptions

| Feature | Range | Data Type | Business Meaning |
|---|---|---|---|
| `Age` | 18–59 | int | Customer age |
| `Balance` | 1,000–99,999 | int | Current account balance |
| `Salary` | 20,000–149,999 | int | Annual income |
| `Tenure` | 1–9 | int | Years as a customer |
| `NumProducts` | 1–3 | int | Products held with bank |
| `IsActive` | 0 or 1 | int | Whether account is active |
| `Churn` | 0 or 1 | int | **Target**: 1 = churned |

---

## 3. Preprocessing Pipeline

### Feature/Target Split

```python
X = data.drop('Churn', axis=1)   # Shape: (1000, 6)
y = data['Churn']                  # Shape: (1000,)
```

### Feature Scaling

`StandardScaler` is applied to normalize all features to zero mean and unit variance. This is critical for distance-based models (KNN, SVM) and regularized models (Logistic Regression).

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

> **Important:** The scaler is fit on the full `X` before the train/test split — this is intentional for cross-validation compatibility. In strict production use, fit the scaler only on training data.

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

- Training set: 800 samples (80%)
- Test set: 200 samples (20%)

---

## 4. Model Training

Five scikit-learn classifiers are instantiated with default hyperparameters and trained on `X_train`:

```python
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest":        RandomForestClassifier(),
    "Decision Tree":        DecisionTreeClassifier(),
    "KNN":                  KNeighborsClassifier(),
    "SVM":                  SVC(probability=True)
}
```

> `SVC(probability=True)` is required to enable `.predict_proba()` for ROC-AUC calculation.

### Cross-Validation

Before final evaluation, each model is assessed via 5-fold cross-validation on the full scaled dataset:

```python
cv_score = cross_val_score(model, X_scaled, y, cv=5).mean()
```

This provides a more stable generalization estimate than a single train/test split.

---

## 5. Evaluation Framework

### `evaluate()` Function

```python
def evaluate(y_test, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob)
    }
```

### Metric Definitions

| Metric | Formula | What it measures |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Quality of positive predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positives |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance between precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across thresholds |

### Confusion Matrix

Generated for each model:

```python
confusion_matrix(y_test, y_pred)
```

Outputs a 2×2 matrix:
```
[[TN  FP]
 [FN  TP]]
```

---

## 6. Visualizations

### ROC Curve

Plots the True Positive Rate vs False Positive Rate for all models on the same axes. A curve closer to the top-left corner indicates a better model.

```python
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=model_name)
```

### Model Comparison Bar Chart

A grouped bar chart comparing `Test Accuracy`, `F1 Score`, and `ROC-AUC` across all five models.

### Feature Importance (Random Forest)

Horizontal bar chart showing the relative importance of each feature as computed by the Random Forest's mean decrease in impurity:

```python
importances = rf.feature_importances_
plt.barh(feature_names, importances)
```

Since `Churn = (Balance > 50000) & (IsActive == 0)`, `Balance` and `IsActive` are expected to dominate feature importances.

---

## 7. Inference Pipeline

The `inference.py` script provides a self-contained, production-ready prediction function. It retrains the best model (Random Forest) on the full dataset and exposes a `predict_churn()` function.

### Usage

```python
from inference import predict_churn

customer = {
    "Age": 35,
    "Balance": 75000,
    "Salary": 60000,
    "Tenure": 3,
    "NumProducts": 1,
    "IsActive": 0
}

result = predict_churn(customer)
print(result)
# {'prediction': 1, 'churn_probability': 0.87, 'label': 'Churn'}
```

### Input Schema

All six features are required as a Python dictionary with integer values:

| Key | Type | Valid Range |
|---|---|---|
| `Age` | int | 18–59 |
| `Balance` | int | 1,000–100,000 |
| `Salary` | int | 20,000–150,000 |
| `Tenure` | int | 1–9 |
| `NumProducts` | int | 1–3 |
| `IsActive` | int | 0 or 1 |

### Output Schema

```python
{
    "prediction": int,          # 0 = No Churn, 1 = Churn
    "churn_probability": float, # Probability score (0.0 – 1.0)
    "label": str                # "Churn" or "No Churn"
}
```

---

## 8. Results Interpretation

Given the deterministic churn rule (`Balance > 50000 AND IsActive == 0`), tree-based models (Random Forest, Decision Tree) are expected to achieve near-perfect accuracy since they can learn exact threshold splits.

Linear models (Logistic Regression, SVM) may score slightly lower because the churn boundary is a non-linear AND condition in the original feature space, though StandardScaler helps.

KNN performance depends heavily on the local density of the feature space.

**Key insight from Feature Importance:** `Balance` and `IsActive` will almost always be the top-2 features since the target is a direct function of these two variables.

---

## 9. Extending the Project

### Use a Real Dataset

Replace the synthetic data block with a Kaggle dataset such as the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset:

```python
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### Add Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

### Handle Class Imbalance

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
```

### Save and Load Models

```python
import joblib

# Save
joblib.dump(rf, "random_forest_churn.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load
model = joblib.load("random_forest_churn.pkl")
scaler = joblib.load("scaler.pkl")
```

---

## 10. Dependencies Reference

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24.0 | Numerical arrays, random data generation |
| `pandas` | ≥2.0.0 | DataFrame operations, results display |
| `matplotlib` | ≥3.7.0 | ROC curves, bar charts, feature importance |
| `scikit-learn` | ≥1.3.0 | All ML models, preprocessing, metrics |
| `jupyter` | ≥1.0.0 | Notebook environment |
| `notebook` | ≥7.0.0 | Jupyter notebook server |
| `ipykernel` | ≥6.25.0 | Python kernel for Jupyter |
