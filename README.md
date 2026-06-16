# Customer Churn Prediction

A machine learning project that trains, evaluates, and compares multiple classification models to predict customer churn based on banking/financial customer attributes.

---

## Overview

Customer churn — when a customer stops using a service — is a critical business metric. This project simulates a realistic bank customer dataset and benchmarks five classification algorithms to identify which customers are most likely to churn, enabling proactive retention strategies.

**Churn Logic:** A customer is labeled as churned if their `Balance > 50,000` AND `IsActive == 0`.

---

## Features

- Synthetic dataset generation with configurable size (default: 1,000 customers)
- StandardScaler-based feature normalization
- 5-fold cross-validation for all models
- Side-by-side comparison of 5 classifiers
- ROC curve visualization
- Random Forest feature importance analysis
- Clean inference script for production use

---

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble, provides feature importances |
| Decision Tree | Interpretable tree-based model |
| K-Nearest Neighbors | Distance-based classifier |
| Support Vector Machine | Kernel-based, `probability=True` |

---

## Dataset

The dataset is synthetically generated using `numpy.random`. No external data download is required.

| Feature | Type | Description |
|---|---|---|
| `Age` | int | Customer age (18–59) |
| `Balance` | int | Account balance (1,000–100,000) |
| `Salary` | int | Annual salary (20,000–150,000) |
| `Tenure` | int | Years with bank (1–9) |
| `NumProducts` | int | Number of products held (1–3) |
| `IsActive` | int | Active customer flag (0 or 1) |
| `Churn` | int | **Target** — 1 = churned, 0 = retained |

---

## Project Structure

```
customer-churn-prediction/
│
├── Costumer_churn_prediction.ipynb   # Original exploratory notebook
├── inference.py                      # Standalone prediction script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── documentation.md                  # Full technical documentation
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook Costumer_churn_prediction.ipynb
```

### 4. Run inference on new data

```bash
python inference.py
```

---

## Evaluation Metrics

Each model is assessed on:

- **CV Accuracy** — 5-fold cross-validation mean accuracy
- **Test Accuracy** — Held-out test set (20% split)
- **Precision** — Of predicted churners, how many actually churned
- **Recall** — Of actual churners, how many were caught
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the ROC curve

---

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib

See `requirements.txt` for pinned versions.

---

## License

MIT License. Free to use and modify.
