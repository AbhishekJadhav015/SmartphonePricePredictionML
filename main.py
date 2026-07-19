import sys
import mlflow
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException 

def start_training_pipeline():
    try:
        
        logging.info("Starting the training pipeline...")
        print("--- Step 1: Initiating Data Ingestion ---")
        
        # Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
        
        # Data Transformation
        logging.info("Starting data transformation...")
        print("--- Step 2: Initiating Data Transformation ---")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"Data transformation completed. Preprocessor path: {preprocessor_path}")
        
        # Model Training
        logging.info("Starting model training...")
        print("--- Step 3: Initiating Model Training ---")
        model_trainer = ModelTrainer()
        
        # score 
        model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed. Model R2 Score: {model_score}")
        
        print(f"\nTraining pipeline completed successfully! Model R2 Score: {model_score}")
        print("Artifacts safely generated inside the 'artifacts/' directory.")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    start_training_pipeline()
