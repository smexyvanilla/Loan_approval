import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "K-Neighbors": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(eval_metric="logloss"),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [4, 6, 8, None],
                },
                "Random Forest": {
                    "n_estimators": [64, 128, 256],
                    "max_depth": [6, 10, None],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [64, 128],
                },
                "AdaBoost": {
                    "learning_rate": [0.05, 0.1, 1.0],
                    "n_estimators": [64, 128],
                },
                "K-Neighbors": {
                    "n_neighbors": [5, 7, 9, 11],
                },
                "XGBClassifier": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [64, 128],
                    "max_depth": [3, 5],
                },
            }

            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models, param=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Model scores (ROC-AUC): {model_report}")

            if best_model_score < 0.6:
                raise CustomException("No reasonably good model found", sys)

            logging.info(f"Best model: {best_model_name} with ROC-AUC {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            acc = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)
            cm = confusion_matrix(y_test, predicted)

            logging.info(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            logging.info(f"Confusion matrix:\n{cm}")

            return best_model_name, best_model_score, acc, f1

        except Exception as e:
            raise CustomException(e, sys)
