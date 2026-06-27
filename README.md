# Credit Risk Loan Approval Predictor

An end-to-end machine learning project that predicts whether a personal loan application should be **Approved** or **Rejected**, based on the applicant's financial and credit profile. Built with a synthetic but realistic Indian-context dataset and deployed as a Flask web app.

🔗 **Live demo:** https://loan-approval-project-nqkr.onrender.com/
---

## Overview

Given an applicant's details — age, income, CIBIL score, employment type, existing loans, requested loan amount, etc. — the model predicts the loan decision along with a confidence score.

This is a binary classification problem:
- **Approved**
- **Rejected**

---

## Dataset

The dataset (`notebook/data/credit_risk.csv`) is synthetically generated (`notebook/generate_data.py`) using real-world credit risk underwriting logic rather than random labels. The approval label is derived from a scoring rule built on:

- CIBIL score (dominant factor, 300–900)
- Monthly income & employment type (Salaried / Self-Employed / Business Owner / Unemployed)
- Debt-to-income ratio (calculated from requested loan amount, term, and income)
- Existing loans count
- Credit history length
- Education, residence type, city tier, and dependents

Gaussian noise is added on top so the relationship is realistic and learnable rather than a clean rule — similar to how real bank approval data behaves.

**Features used:**

| Feature | Description |
|---|---|
| age | Applicant's age |
| gender | Male / Female |
| marital_status | Single / Married |
| dependents | Number of dependents |
| education | Graduate / Not Graduate |
| employment_type | Salaried / Self-Employed / Business Owner / Unemployed |
| city_tier | Tier 1 / Tier 2 / Tier 3 |
| residence_type | Owned / Rented / Living with Parents |
| monthly_income | Monthly income in INR |
| cibil_score | Credit score (300–900) |
| existing_loans_count | Number of active loans |
| credit_history_length_years | Years of credit history |
| bank_balance | Current bank balance in INR |
| loan_amount | Requested loan amount in INR |
| loan_term_months | Requested loan tenure |
| loan_purpose | Home Renovation / Education / Medical / Wedding / Vehicle / Business Expansion / Debt Consolidation / Travel |
| debt_to_income_ratio | Derived feature: annual EMI ÷ annual income |

---

## Project Structure

```
├── app.py                          # Flask application
├── render.yaml                     # Render deployment config
├── Procfile                         # Process file for deployment
├── runtime.txt                      # Python version pin
├── requirements.txt
├── setup.py
├── notebook/
│   ├── generate_data.py            # Synthetic dataset generator
│   └── data/credit_risk.csv
├── artifacts/                       # Saved model, preprocessor, train/test splits
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form + result
└── src/
    ├── components/
    │   ├── data_ingestion.py       # Reads data, train/test split
    │   ├── data_transformation.py  # Preprocessing pipeline (scaling + one-hot encoding)
    │   └── model_trainer.py        # Trains & compares multiple classifiers
    ├── pipeline/
    │   ├── train_pipeline.py       # Orchestrates the training pipeline
    │   └── predict_pipeline.py     # CustomData + PredictPipeline for inference
    ├── logger.py
    ├── exception.py
    └── utils.py
```

---

## Model

Several classifiers are trained and compared using `GridSearchCV`, with the best one selected by ROC-AUC on the held-out test set:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- XGBoost

**Best model:** Logistic Regression
**ROC-AUC:** 0.95
**Accuracy:** 0.88
**F1 Score:** 0.91

The trained model and preprocessing pipeline are saved as `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.


## Tech Stack

Python · Flask · Scikit-learn · XGBoost · Pandas · NumPy · Gunicorn

---

## Disclaimer

This project uses **synthetic data** for educational/portfolio purposes. It is not trained on real applicant data and should not be used for actual lending decisions.
