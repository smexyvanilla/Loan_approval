import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


NUMERICAL_COLUMNS = [
    "age",
    "dependents",
    "monthly_income",
    "cibil_score",
    "existing_loans_count",
    "credit_history_length_years",
    "bank_balance",
    "loan_amount",
    "loan_term_months",
    "debt_to_income_ratio",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "city_tier",
    "residence_type",
    "loan_purpose",
]

TARGET_COLUMN = "loan_approved"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Builds the preprocessing ColumnTransformer for numerical + categorical
        credit-risk features.
        """
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            logging.info(f"Numerical columns: {NUMERICAL_COLUMNS}")
            logging.info(f"Categorical columns: {CATEGORICAL_COLUMNS}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, NUMERICAL_COLUMNS),
                    ("cat_pipeline", cat_pipeline, CATEGORICAL_COLUMNS),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].map({"Rejected": 0, "Approved": 1})

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].map({"Rejected": 0, "Approved": 1})

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
