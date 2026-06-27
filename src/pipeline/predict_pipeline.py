import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred_proba = model.predict_proba(data_scaled)[:, 1]
            pred_label = (pred_proba >= 0.5).astype(int)

            return pred_label, pred_proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps raw form inputs to a one-row DataFrame matching the training schema.
    """

    def __init__(
        self,
        age: int,
        gender: str,
        marital_status: str,
        dependents: int,
        education: str,
        employment_type: str,
        city_tier: str,
        residence_type: str,
        monthly_income: float,
        cibil_score: int,
        existing_loans_count: int,
        credit_history_length_years: float,
        bank_balance: float,
        loan_amount: float,
        loan_term_months: int,
        loan_purpose: str,
    ):
        self.age = age
        self.gender = gender
        self.marital_status = marital_status
        self.dependents = dependents
        self.education = education
        self.employment_type = employment_type
        self.city_tier = city_tier
        self.residence_type = residence_type
        self.monthly_income = monthly_income
        self.cibil_score = cibil_score
        self.existing_loans_count = existing_loans_count
        self.credit_history_length_years = credit_history_length_years
        self.bank_balance = bank_balance
        self.loan_amount = loan_amount
        self.loan_term_months = loan_term_months
        self.loan_purpose = loan_purpose

    def get_debt_to_income_ratio(self):
        """Same approximate EMI logic used when generating training data."""
        annual_income = self.monthly_income * 12
        r = 0.12 / 12
        n = self.loan_term_months
        emi = self.loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
        return round((emi * 12) / annual_income, 3)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "marital_status": [self.marital_status],
                "dependents": [self.dependents],
                "education": [self.education],
                "employment_type": [self.employment_type],
                "city_tier": [self.city_tier],
                "residence_type": [self.residence_type],
                "monthly_income": [self.monthly_income],
                "cibil_score": [self.cibil_score],
                "existing_loans_count": [self.existing_loans_count],
                "credit_history_length_years": [self.credit_history_length_years],
                "bank_balance": [self.bank_balance],
                "loan_amount": [self.loan_amount],
                "loan_term_months": [self.loan_term_months],
                "loan_purpose": [self.loan_purpose],
                "debt_to_income_ratio": [self.get_debt_to_income_ratio()],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
