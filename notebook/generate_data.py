"""
Generates a synthetic but realistic Indian-context personal loan / credit-risk
dataset. The label (loan_approved) is produced from an underwriting-style
scoring rule built from real credit risk factors (CIBIL score, income, DTI,
employment stability, existing loans, etc.) plus noise, NOT pure randomness -
so the dataset has genuine, learnable signal, similar to how a real bank
dataset would behave.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 6000

genders = np.random.choice(["Male", "Female"], size=N, p=[0.58, 0.42])
marital_status = np.random.choice(["Single", "Married"], size=N, p=[0.42, 0.58])
education = np.random.choice(
    ["Graduate", "Not Graduate"], size=N, p=[0.62, 0.38]
)
employment_type = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    size=N, p=[0.52, 0.22, 0.18, 0.08]
)
city_tier = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=N, p=[0.35, 0.4, 0.25])
residence_type = np.random.choice(["Owned", "Rented", "Living with Parents"], size=N, p=[0.38, 0.42, 0.2])

age = np.clip(np.random.normal(35, 9, N), 21, 65).astype(int)

dependents = np.clip(np.random.poisson(1.3, N), 0, 5)

# Monthly income (INR), employment-type and city-tier dependent
base_income = np.random.lognormal(mean=10.6, sigma=0.55, size=N)
income_multiplier = np.select(
    [employment_type == "Salaried", employment_type == "Self-Employed",
     employment_type == "Business Owner", employment_type == "Unemployed"],
    [1.0, 1.1, 1.25, 0.25]
)
city_multiplier = np.select(
    [city_tier == "Tier 1", city_tier == "Tier 2", city_tier == "Tier 3"],
    [1.25, 1.0, 0.8]
)
monthly_income = np.round(base_income * income_multiplier * city_multiplier / 1000) * 1000
monthly_income = np.clip(monthly_income, 8000, 500000)

# CIBIL score (300-900), correlated with age (credit history) and income, with noise
cibil_score = (
    600
    + (age - 21) * 3.2
    + (np.log1p(monthly_income) - 9) * 18
    - dependents * 4
    + np.where(employment_type == "Unemployed", -60, 0)
    + np.random.normal(0, 45, N)
)
cibil_score = np.clip(cibil_score, 300, 900).astype(int)

existing_loans_count = np.clip(np.random.poisson(0.9, N), 0, 6)

loan_amount = np.round(np.random.lognormal(mean=12.3, sigma=0.6, size=N) / 1000) * 1000
loan_amount = np.clip(loan_amount, 20000, 4000000)

loan_term_months = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], size=N,
                                     p=[0.08, 0.12, 0.15, 0.13, 0.17, 0.1, 0.1, 0.08, 0.07])

loan_purpose = np.random.choice(
    ["Home Renovation", "Education", "Medical", "Wedding", "Vehicle",
     "Business Expansion", "Debt Consolidation", "Travel"],
    size=N
)

annual_income = monthly_income * 12
# Approx EMI using simple interest assumption for feature purposes (not exact amortization)
assumed_annual_rate = 0.12
r = assumed_annual_rate / 12
emi = loan_amount * r * (1 + r) ** loan_term_months / ((1 + r) ** loan_term_months - 1)
debt_to_income_ratio = np.round((emi * 12) / annual_income, 3)

credit_history_length_years = np.clip((age - 21) * np.random.uniform(0.3, 0.9, N), 0, 35).round(1)

bank_balance = np.round(monthly_income * np.random.uniform(0.5, 6, N) / 1000) * 1000

# ---- Underwriting-style approval score (the "real factors" logic) ----
score = (
    0.0
    + (cibil_score - 650) * 0.018          # CIBIL is the dominant factor
    + (np.log1p(monthly_income) - 10) * 1.1
    - debt_to_income_ratio * 3.2
    - existing_loans_count * 0.35
    + np.where(employment_type == "Salaried", 0.5, 0)
    + np.where(employment_type == "Business Owner", 0.2, 0)
    + np.where(employment_type == "Unemployed", -2.5, 0)
    + np.where(education == "Graduate", 0.25, 0)
    + credit_history_length_years * 0.03
    - dependents * 0.08
    + np.where(residence_type == "Owned", 0.3, 0)
    + np.random.normal(0, 0.9, N)           # noise so it's not a clean separable rule
)

prob_approved = 1 / (1 + np.exp(-score))
loan_approved = (prob_approved > 0.5).astype(int)
loan_approved = np.where(loan_approved == 1, "Approved", "Rejected")

df = pd.DataFrame({
    "age": age,
    "gender": genders,
    "marital_status": marital_status,
    "dependents": dependents,
    "education": education,
    "employment_type": employment_type,
    "city_tier": city_tier,
    "residence_type": residence_type,
    "monthly_income": monthly_income.astype(int),
    "cibil_score": cibil_score,
    "existing_loans_count": existing_loans_count,
    "credit_history_length_years": credit_history_length_years,
    "bank_balance": bank_balance.astype(int),
    "loan_amount": loan_amount.astype(int),
    "loan_term_months": loan_term_months,
    "loan_purpose": loan_purpose,
    "debt_to_income_ratio": debt_to_income_ratio,
    "loan_approved": loan_approved,
})

out_path = os.path.join(os.path.dirname(__file__), "data", "credit_risk.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["loan_approved"].value_counts(normalize=True))
