import logging
import numpy as np
import pandas as pd
import os 
import sys
import pickle 
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Utility functions for saving objects 
def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj :
            pickle.dump(obj ,file_obj)
            
    except Exception as e :
        raise CustomException(e,sys)
    
# Utility function for evaluating models
def evaluate_model(X_train, y_train,X_test,y_test,models,param):
    try:
        model_report = {}
        trained_models = {}

        # Loop through each item in your dictionary
        for model_name, model_obj in models.items():
            try:
                logging.info(f"Tuning hyperparameters for: {model_name}")
                model_param = param.get(model_name, {})

                # Execute your Cross-Validation Grid Search
                gs = GridSearchCV(model_obj, model_param, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)

                # Assign the absolute best parameters found
                best_model = gs.best_estimator_
                best_model.fit(X_train, y_train)

                # Compute your performance matrix
                y_test_pred = best_model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)

                # POPULATE BOTH OUTPUT DICTIONARIES Explicitly
                model_report[model_name] = test_model_score
                trained_models[model_name] = best_model

            except Exception as single_model_error:
                # If one specific model fails, log it and keep testing the others!
                logging.warning(f"Model {model_name} failed to evaluate. Error: {single_model_error}")
                continue

        if not model_report:
            raise Exception("All models failed during the evaluation process.")
            
        return model_report, trained_models

    except Exception as e:
        raise CustomException(e, sys)

# Utility function for loading objects  
def load_obj(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e ,sys)
    