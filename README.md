# Telecom Customer Churn Prediction

**University of Mines and Technology (UMaT), Tarkwa, Ghana**
BSc Statistical Data Science — Final Year Project, April 2026

**Team:** Benedicta Amoah · Delle Micheal · Arthur Lordys Nana Ama Kissiwaa · Essah Isaac
**Supervisor:** Assoc. Prof. Peter Kwesi Nyarko

---

## Overview

Customer churn — when a subscriber leaves a telecom provider — directly erodes revenue and raises acquisition costs. This project develops and comparatively evaluates machine learning models to predict churn, identify its key drivers, and recommend data-driven retention strategies.

Five supervised classifiers are trained and benchmarked across two real-world telecom datasets:

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Rule-based |
| Random Forest | Ensemble (bagging) |
| Support Vector Machine | Kernel-based |
| XGBoost | Ensemble (gradient boosting) |

---

## Datasets

| Dataset | Rows | Features | Churn Rate | Source |
|---|---|---|---|---|
| `Telco_customer_churn.xlsx` | 7,043 | 33 | 26.5% | IBM-inspired US telecom |
| `indian_telecom_customers.csv` | 15,000 | 39 | 49.0% | Indian market simulation |

Both datasets are stored in [`data/raw/`](data/raw/).

---

## Project Structure

```
teleco-churn-rate-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned, encoded, split data (train/val/test)
├── notebooks/
│   ├── 01_eda.ipynb            # Milestone 1 — Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Milestone 2 — Preprocessing & Feature Engineering
│   ├── 03_modeling.ipynb       # Milestone 3 — Model Training (GridSearchCV)
│   └── 04_evaluation.ipynb     # Milestone 4 — Evaluation, ROC, Feature Importance
├── models/                     # Saved model (.pkl) and scaler files
├── reports/
│   └── figures/                # All exported plots (20 figures)
├── src/                        # Reusable Python modules
├── requirements.txt
└── README.md
```

---

## Methodology

### Preprocessing
- Dropped PII, leakage, and high-cardinality geography columns
- Fixed `Total Charges` dtype (whitespace → NaN → 0 for zero-tenure rows)
- **Feature engineering:** tenure group (Short/Medium/Long), service count, charge-per-month ratio
- **Encoding:** One-Hot Encoding for all categorical variables
- **Scaling:** Z-score standardisation (fit on training set only)
- **Data split:** 70% train / 15% validation / 15% test (stratified)
- **Class imbalance:** SMOTE applied to training set only

### Model Training
- GridSearchCV with 3-fold cross-validation, F1-score as the tuning metric
- Hyperparameters tuned per model (C, max_depth, n_estimators, learning_rate, etc.)
- All models trained on SMOTE-balanced training set; evaluated on original-distribution val/test sets

### Evaluation Metrics
Accuracy · Precision · Recall · F1-Score · ROC-AUC

---

## Results

### Telco Dataset — Test Set

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.7531 | 0.5220 | **0.8036** | **0.6329** | 0.8535 |
| **XGBoost** | 0.7796 | 0.5693 | 0.6893 | 0.6236 | **0.8559** |
| Decision Tree | 0.7682 | 0.5499 | 0.6893 | 0.6117 | 0.8256 |
| SVM | 0.7711 | 0.5565 | 0.6679 | 0.6071 | 0.8159 |
| Random Forest | **0.7852** | **0.5930** | 0.6036 | 0.5982 | 0.8330 |

Logistic Regression leads on F1 and Recall (catches 80% of actual churners). XGBoost leads on ROC-AUC, making it the stronger risk-scoring model.

### India Dataset — Test Set

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.6951** | 0.6901 | 0.6863 | **0.6882** | **0.7695** |
| **Random Forest** | 0.6911 | 0.6835 | 0.6890 | 0.6862 | 0.7642 |
| Logistic Regression | 0.6720 | 0.6605 | **0.6809** | 0.6705 | 0.7380 |
| SVM | 0.6707 | 0.6622 | 0.6700 | 0.6661 | 0.7335 |
| Decision Tree | 0.6813 | **0.6900** | 0.6355 | 0.6616 | 0.7373 |

XGBoost dominates all metrics on the India dataset. Random Forest is nearly identical, confirming ensemble superiority for this domain.

---

## Key Churn Drivers

**Telco Dataset**
1. **Contract type (Month-to-Month)** — 42% churn vs <10% for 1- or 2-year contracts
2. **Tenure** — Customers in their first 12 months are at extreme risk
3. **Monthly charges** — Higher billing correlates strongly with churn
4. **Internet service (Fiber Optic)** — 42% churn rate vs 19% DSL
5. **Payment method (Electronic Check)** — Highest churn among all payment types

**India Dataset**
1. **Overall satisfaction score** — Strongest single predictor; low scores directly precede churn
2. **Price sensitivity score** — High-sensitivity customers leave proactively
3. **Competitor offers received** — 3+ offers received sharply increases churn probability
4. **Support calls last month** — High call frequency signals service dissatisfaction
5. **Tenure** — New subscribers (< 12 months) most vulnerable

---

## Retention Strategy Recommendations

1. **Early Warning System** — Deploy XGBoost to score all customers monthly; trigger retention workflows for the top 20% risk tier
2. **New Customer Onboarding** — Dedicated relationship check-ins at months 3, 6, and 9 for all new subscribers
3. **Contract Migration Campaign** — Incentivise month-to-month subscribers with a discounted 1-year upgrade offer
4. **Satisfaction Recovery SLA** — Customers with satisfaction ≤ 4 or multiple support calls receive priority resolution within 24 hours
5. **Competitor Response Protocol** — Trigger a personalised retention offer within 48 hours of detecting 2+ competitor offers received
6. **Auto-Pay Incentive** — Offer 5% monthly discount to electronic-check customers who switch to automatic payment

---

## How to Run

```bash
# 1. Clone / navigate to the project
cd teleco-churn-rate-prediction

# 2. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the Jupyter kernel
python3 -m ipykernel install --user --name=telco-churn --display-name="Python 3 (telco-churn)"

# 5. Launch Jupyter and run notebooks in order
jupyter notebook notebooks/
```

Run the four notebooks in sequence:

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `01_eda.ipynb` | Understand the data |
| 2 | `02_preprocessing.ipynb` | Prepare data, apply SMOTE |
| 3 | `03_modeling.ipynb` | Train all 5 models |
| 4 | `04_evaluation.ipynb` | Evaluate on test set, generate plots |

---

## Dependencies

| Package | Version |
|---|---|
| pandas | 3.0.3 |
| numpy | ≥ 1.26 |
| scikit-learn | ≥ 1.9 |
| xgboost | ≥ 3.2 |
| imbalanced-learn | ≥ 0.14 |
| matplotlib | ≥ 3.10 |
| seaborn | ≥ 0.13 |

See [requirements.txt](requirements.txt) for the full list.
