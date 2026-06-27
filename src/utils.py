import os
import sys
import dill

from src.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    """
    Grid-searches each model over its param grid and scores it on the test
    set using ROC-AUC (good metric for imbalanced approve/reject classes).
    Returns a dict: {model_name: test_roc_auc}
    """
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(x_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)

            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(x_test)[:, 1]
                test_score = roc_auc_score(y_test, y_test_proba)
            else:
                test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
