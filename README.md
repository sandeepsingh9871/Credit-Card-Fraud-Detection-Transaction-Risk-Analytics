# Credit Card Fraud Detection — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?logo=mysql)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Tableau](https://img.shields.io/badge/Tableau-Desktop-blue?logo=tableau)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.977-brightgreen)
![F1](https://img.shields.io/badge/F1%20Score-0.849-brightgreen)

> **End-to-end fraud detection pipeline on 284,807 real credit card transactions — from SQL data sourcing through XGBoost modelling, SHAP explainability, business ROI analysis, and an interactive Tableau monitoring dashboard.**

---

## Project Summary

| Metric | Value |
|--------|-------|
| Dataset | 284,807 transactions, 492 fraud cases |
| Fraud rate | 0.1727% (577:1 imbalance) |
| Model | XGBoost + scale_pos_weight + GridSearchCV |
| AUC-ROC | **0.977** |
| AUC-PR | **0.880** |
| Precision | **0.813** |
| Recall | **0.888** |
| F1 Score | **0.849** |
| Fraud caught (TP) | 87 / 98 (88.8%) |
| Fraud missed (FN) | 11 |
| Total business cost | **$1,950** (FP + FN) |
| Fraud value prevented | **$13,050** |
| Net saving vs no model | **$12,750** |

---

## Pipeline Overview

```
Raw Data (Kaggle CSV)
        ↓
Phase 1 — MySQL SQL Analysis
        ↓
Phase 2 — Python EDA (8 charts)
        ↓
Phase 3 — Feature Engineering (8 new features)
        ↓
Phase 4 — Class Imbalance (SMOTE + threshold tuning)
        ↓
Phase 5 — XGBoost Modelling (GridSearchCV)
        ↓
Phase 6 — SHAP Explainability (6 charts)
        ↓
Phase 7 — Business ROI & Cost Analysis
        ↓
Phase 8 — Tableau Interactive Dashboard
```

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 2 days
- Features V1–V28: PCA-anonymised for privacy
- `Amount`: transaction amount in USD
- `Class`: 0 = legitimate, 1 = fraud
- **No missing values**

---

## Phase 1 — SQL Analysis (MySQL)

Loaded raw CSV into MySQL and explored fraud patterns using pure SQL before any Python modelling.

**5 key queries written:**

```sql
-- Fraud rate by hour of day
SELECT
    FLOOR(Time / 3600) MOD 24     AS hour_of_day,
    COUNT(*)                       AS total_txns,
    SUM(Class)                     AS fraud_count,
    ROUND(SUM(Class)*100.0/COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;
```

**SQL findings:**
- 577:1 class imbalance — fraud is 0.17% of all transactions
- Fraud rate peaks at **2am (highest fraud concentration)**
- 70%+ of fraud transactions are under $100 — card-testing pattern
- `CASE WHEN` amount bucketing confirmed card-testing in $10–$99 range
- Window functions (`COUNT(*) OVER PARTITION BY`) used for velocity approximation

**3 reusable views created:**
- `clean_transactions` — enriched data feed for Python
- `fraud_hourly_summary` — hourly fraud rates
- `fraud_amount_buckets` — fraud by amount band

---

## Phase 2 — Exploratory Data Analysis

**8 charts produced:**

| Chart | Key Finding |
|-------|------------|
| Class imbalance | 577:1 ratio — accuracy is meaningless |
| Fraud by hour | Peak at 2–4am — low monitoring window |
| Fraud by time of day | Night (0–5am) is highest risk period |
| Amount distributions | Fraud concentrates under $100 |
| Log-amount transform | Right skew removed — better for models |
| Feature correlations | V14, V17, V12 strongest fraud predictors |
| KDE separation | Good distribution gap in top features |
| Correlation heatmap | PCA features orthogonal — no multicollinearity |

**Key insight:** V14 correlation with fraud = -0.30, V17 = -0.33 — confirmed later by SHAP

---

## Phase 3 — Feature Engineering

Built 8 new features on top of 28 PCA features:

| Feature | Type | Rationale |
|---------|------|-----------|
| `is_night` | Binary | Hours 0–5am — peak fraud window |
| `amt_zscore` | Continuous | Flags extreme amounts (card-testing + large fraud) |
| `amt_bucket` | Ordinal (0–5) | Card-testing pattern in $10–$99 range |
| `txn_velocity_1h` | Continuous | Transaction count in 1-hour window |
| `txn_velocity_10m` | Continuous | Transaction count in 10-minute window |
| `rapid_repeat` | Binary | 3+ transactions in 10 minutes |
| `time_sin` | Cyclical | Sin encoding — hour 23 and 0 are neighbours |
| `time_cos` | Cyclical | Cos encoding — completes cyclical representation |

**Why cyclical encoding?**
Without it, the model treats hour 23 and hour 0 as 23 units apart — but they are only 1 hour apart. Sin/cos encoding preserves this circular relationship.

---

## Phase 4 — Class Imbalance Handling

**Problem:** 577:1 imbalance. Naive model achieves 99.83% accuracy by predicting all-legitimate — but catches zero fraud.

**Strategy comparison:**

| Strategy | AUC-PR | F1 | Approach |
|----------|--------|-----|----------|
| Naive (all legit) | 0.002 | 0.000 | Baseline — useless |
| scale_pos_weight | **0.880** | **0.859** | XGBoost native — winner |
| SMOTE | 0.858 | 0.851 | Synthetic oversampling |

**Key decisions:**
- SMOTE applied on **training data only** — never on test (prevents data leakage)
- `scale_pos_weight = 577.3` (negatives ÷ positives)
- Threshold tuned using Precision-Recall curve — not fixed at 0.5
- Final threshold: **0.09** (cost-optimised, not F1-optimised)

---

## Phase 5 — Modelling

**3-model progression:**

| Model | AUC-ROC | AUC-PR | F1 | Notes |
|-------|---------|--------|-----|-------|
| Logistic Regression | baseline | baseline | baseline | Floor to beat |
| Random Forest | ~0.96 | ~0.84 | ~0.83 | Good, slower |
| XGBoost (default) | 0.976 | 0.878 | 0.848 | Best |
| **XGBoost (tuned)** | **0.977** | **0.880** | **0.849** | **Final model** |

**GridSearchCV parameters tuned:**
```python
param_grid = {
    'max_depth'        : [4, 6, 8],
    'learning_rate'    : [0.05, 0.1, 0.2],
    'n_estimators'     : [100, 200],
    'subsample'        : [0.8, 1.0],
    'colsample_bytree' : [0.8, 1.0],
}
# Scored on: average_precision (AUC-PR)
# CV: StratifiedKFold(n_splits=5)
```

---

## Phase 6 — SHAP Explainability

**Why SHAP?** TreeExplainer gives exact Shapley values for XGBoost — mathematically rigorous attribution from game theory.

**6 charts produced:**

| Chart | Purpose |
|-------|---------|
| Summary plot (beeswarm) | Global feature importance |
| Bar plot | Mean \|SHAP\| ranking |
| Waterfall — fraud transaction | Why THIS transaction was flagged |
| Waterfall — legit transaction | Why THIS transaction was cleared |
| Dependence plots (top 3) | How feature value affects fraud risk |
| Fraud vs legit comparison | Feature importance split by class |

**Top 5 SHAP features:**

| Feature | Mean \|SHAP\| | Direction |
|---------|-------------|-----------|
| V14 | Highest | Negative values → fraud risk ↑ |
| V17 | High | Negative values → fraud risk ↑ |
| V12 | High | Negative values → fraud risk ↑ |
| V10 | Medium | Negative values → fraud risk ↑ |
| txn_velocity_1h | Medium | High velocity → fraud risk ↑ |

**Key insight:** V14 and V17 topped both the correlation analysis (Phase 2) and SHAP rankings — consistent signal across the entire pipeline.

---

## Phase 7 — Business ROI & Cost Analysis

**Cost matrix:**

| Decision | Cost | Business impact |
|----------|------|----------------|
| False Negative (missed fraud) | **$150** | Revenue loss |
| False Positive (blocked legit) | **$15** | Customer friction + ops cost |
| True Positive (caught fraud) | $0 | Fraud prevented |
| True Negative (cleared legit) | $0 | Happy customer |

**Scenario comparison:**

| Scenario | Fraud Caught | FN Cost | FP Cost | Total Cost |
|----------|-------------|---------|---------|-----------|
| No model | 0 | $14,700 | $0 | $14,700 |
| Default threshold 0.5 | 81 | $2,550 | $165 | $2,715 |
| **Optimal threshold 0.09** | **87** | **$1,650** | **$300** | **$1,950** |

**Net saving vs no model: $12,750**

**Sensitivity analysis:** Optimal threshold remains robust across cost assumptions — does not flip with small changes in FN/FP cost ratio.

---

## Phase 8 — Tableau Dashboard (LiveDashboard:https://public.tableau.com/views/CreditCardFraudDetectionRiskAnalyticsDashboard/Dashboard1?:language=enUS&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link 

**4 interactive views:**

| View | Chart Type | Business Question |
|------|-----------|------------------|
| Risk Score Distribution | Histogram (log scale) | How well separated is fraud from legit? |
| Fraud Rate by Hour | Dual-axis line + bar | Which hours have highest fraud risk? |
| Business Cost by Outcome | Stacked bar + KPIs | What does each decision type cost? |
| Fraud Investigation Priority Matrix | Highlight table | Which risk tier to investigate first? |

**KPI strip (always visible):**
- Total Business Cost: **$1,950**
- Fraud Cases Caught: **87**
- Fraud Cases Missed: **11** (red)
- Total Fraud Value at Risk: **$1,825.93**

**Interactive features:**
- Click any Risk Band row → filters all charts to that tier
- Outcome color coding: TP=green, FP=amber, FN=red, TN=gray
- Decision threshold reference line at 0.09

---

## Repository Structure

```
credit-card-fraud-detection/
│
├── sql/
│   └── phase1_sql_analysis.sql
│
├── notebooks/
│   ├── phase2_eda.ipynb
│   ├── phase3_feature_engineering.ipynb
│   ├── phase4_class_imbalance.ipynb
│   ├── phase5_modelling.ipynb
│   ├── phase6_shap.ipynb
│   └── phase7_business_roi.ipynb
│
├── data/
│   ├── clean_transactions_full.csv   (MySQL export)
│   ├── featured_transactions.csv     (Phase 3 output)
│   ├── scored_transactions.csv       (Phase 5 output)
│   └── tableau_fraud_data.csv        (Phase 8 input)
│
├── models/
│   ├── fraud_model_xgb.pkl
│   ├── scaler.pkl
│   └── phase4_config.json
│
├── tableau/
│   └── Credit Card Fraud Detection - Risk Analytics Dashboard.twbx
│
├── outputs/
│   └── (all saved .png charts)
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place creditcard.csv in the data/ folder
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Run notebooks in order
# Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
joblib
mysql-connector-python
plotly
```

## Results Summary

```
Dataset        : 284,807 transactions | 492 fraud cases | 0.17% fraud rate
Model          : XGBoost (scale_pos_weight=577 | GridSearchCV | 5-fold stratified CV)
AUC-ROC        : 0.977
AUC-PR         : 0.880
Precision      : 0.813
Recall         : 0.888
F1             : 0.849
Threshold      : 0.09 (cost-optimised)
Fraud caught   : 87 / 98 (88.8% recall)
Business cost  : $1,950 total | $12,750 saved vs no model
Top features   : V14, V17, V12 (SHAP + correlation consistent)
```

---

*Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)*
