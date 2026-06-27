import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

            trainer = ModelTrainer()
            best_model_name, best_score, acc, f1 = trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("Training pipeline completed")

            print(f"Best model: {best_model_name}")
            print(f"ROC-AUC: {best_score:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}")

            return best_model_name, best_score, acc, f1

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
