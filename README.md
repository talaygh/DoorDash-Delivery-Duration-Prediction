# Delivery Duration Prediction

**Python · Scikit-learn · XGBoost · LightGBM · TensorFlow**

---

## Overview

A machine learning project built on historical DoorDash order data to predict actual_total_delivery_duration in seconds, covering feature engineering, multicollinearity analysis, classical ML model comparison, and a deep learning baseline — with a two-stage modeling approach that first predicts prep time and reconstructs total duration.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11.9 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Feature Selection | Scikit-learn (Random Forest, PCA) |
| Multicollinearity | Statsmodels (VIF) |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow / Keras |
| Environment | pyenv, VS Code |

---

## Approach

1. **Data Exploration** — parse timestamps, compute target variable, engineer initial features
2. **Feature Engineering & Encoding** — one-hot encode categoricals, derive ratio/range features to reduce collinearity
3. **Correlation Analysis** — drop or transform highly correlated features
4. **VIF Check** — iteratively remove features with VIF > 20
5. **Feature Selection** — Random Forest Gini importance + PCA explained variance
6. **Classical ML** — compare Ridge, Decision Tree, Random Forest, XGBoost, LightGBM, MLP across multiple feature sets and scalers
7. **Prep-time Decomposition** — predict prep time with LGBM, sum with estimated durations, fit a stage-2 linear model on reconstructed features
8. **Deep Learning** — simple ANN as a baseline comparison

---

## Results

Best performing pipeline: **LightGBM** for prep time prediction → **Linear Regression** stage-2 model on reconstructed features.

---

## Project Structure

```
├── delivery_duration_prediction.ipynb   # Main notebook
├── historical_data.csv                  # Raw data
└── README.md
```

---

## Setup

```bash
# Python 3.11.9 via pyenv
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm statsmodels tensorflow
```

---

*Talay Kamali*
