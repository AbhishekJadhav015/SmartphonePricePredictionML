import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model  
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import r2_score

# Configuration class for model trainer
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

# ModelTrainer class for training and evaluating models    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting train and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
        
            xgb_device = 'cpu'
            cat_task_type = 'CPU'
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": xgb.XGBRegressor(
                    device=xgb_device,              # Changed to 'cpu' for grid compatibility
                    tree_method='hist',             # High-performance histogram method
                    objective='reg:squarederror'
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    task_type=cat_task_type,
                    verbose=False,allow_writing_files=False, 
                    train_dir='artifacts/catboost_info'
                ),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            param = {
                "Linear Regression": {},
                
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                
                "Ridge": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto']
                },
                
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },
                
                "Random Forest Regressor": {  
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [50, 100, 200]
                },
                
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                
                "CatBoosting Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'iterations': [100, 200, 300]
                },
                
                "AdaBoost Regressor": {
                    'learning_rate': [0.05, 0.1, 0.2, 0.5],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [50, 100, 200]
                }
            }
            # Evaluate models and get the best model
            logging.info("Evaluating models")
            model_report, trained_models = evaluate_model(
                X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test, models=models, param=param
            )
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            best_model = trained_models[best_model_name]
            
            # Check if the best model score is below a certain threshold 
            if best_model_score < 0.6:
                raise CustomException("NO best model found")
            logging.info(f"Best model found: {best_model_name} with R2 score of {best_model_score}")            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved")
            
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
