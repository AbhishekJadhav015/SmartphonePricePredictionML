from narwhals import col
import pandas as pd
import re
import numpy as np
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

# Configuration class for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
  
# DataTransformation class for data cleaning and preprocessing  
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def clean_memory_storage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies your custom regex engine to clean Memory and Storage."""
        logging.info("Applying Regex cleaning logic on Memory and Storage features.")
        
        def extract_and_convert(val):
            val = str(val).lower().strip()
            match = re.search(r"(\d+\.?\d*)\s*(gb|mb)", val)
            if match:
                number = float(match.group(1))
                unit = match.group(2)
                return number / 1024 if unit == 'mb' else number
            
            only_number = re.search(r"(\d+\.?\d*)", val)
            return float(only_number.group(1)) if only_number else 0.0

        for col in ['Memory', 'Storage']:
            df[col] = df[col].convert_dtypes().astype(str).str.lower().str.split()
            df[f'{col}_Cleaned'] = df[col].apply(extract_and_convert)
            
        return df
    
    def filter_outliers_and_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters valid ranges and drops invalid rows entirely."""
        logging.info("Filtering valid ranges and removing unauthentic data rows.")
        
        # Enforce range limits
        df['Storage_Cleaned'] = pd.to_numeric(df['Storage_Cleaned'], errors='coerce')
        df['Storage_Cleaned'] = df['Storage_Cleaned'].where(df['Storage_Cleaned'].between(4, 2048), np.nan)
        
        df['Memory_Cleaned'] = pd.to_numeric(df['Memory_Cleaned'], errors='coerce')
        df['Memory_Cleaned'] = df['Memory_Cleaned'].where(df['Memory_Cleaned'].between(0.5, 64), np.nan)
        
        # Drop rows failing the range validation
        df = df.dropna(subset=['Storage_Cleaned', 'Memory_Cleaned'])
        
        # Drop original raw uncleaned source columns
        df = df.drop(columns=["Memory", "Storage"], errors='ignore')
        df.rename(columns={'Original Price': 'Original_Price', 'Storage_Cleaned': 'Storage', 'Memory_Cleaned': 'Memory'}, inplace=True)
        
        return df
    # Method to get the data transformer object
    def get_data_transformer_obj(self):
        
        try:
            numerical_columns =['Memory','Storage','Rating','Original_Price']
            categorical_columns = ['Brand', 'Model', 'Color']
                    
            num_pipline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scalar",StandardScaler())
                ]
            )
            
            cat_pipline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoding",OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )
            logging.info(f"categorical columns:{categorical_columns}")
            logging.info(f"numerical columns:{numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipline",num_pipline,numerical_columns),
                    ("categorical_pipline",cat_pipline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e ,sys)
    
    # Method to initiate data transformation    
    def initiate_data_transformation(self, train_path ,test_path):
            try:
                # Read train and test data
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                
                logging.info("Read train and test data completed")
                
                # Clean and preprocess the data
                for col in ['Brand', 'Model', 'Color']:
                    if col in train_df.columns:
                        train_df[col] = train_df[col].astype(str).str.strip()
                    if col in test_df.columns:
                        test_df[col] = test_df[col].astype(str).str.strip()
                logging.info(f"Rows before dropping target NaNs - Train: {len(train_df)}, Test: {len(test_df)}")
                
                # Drop rows with NaN values in the target column
                train_df = train_df.dropna()
                test_df = test_df.dropna()
                logging.info(f"Rows after dropping target NaNs - Train: {len(train_df)}, Test: {len(test_df)}")
               
                # Clean and filter the data
                train_df = self.clean_memory_storage_columns(train_df)
                train_df = self.filter_outliers_and_ranges(train_df)
                test_df = self.clean_memory_storage_columns(test_df)
                test_df = self.filter_outliers_and_ranges(test_df)
                logging.info("Cleaning and Filtering outliers completed")
                
                # Split features and target variable 
                logging.info("obtaining preprocessing object")
                preprocessing_obj = self.get_data_transformer_obj()
                
                # Split input features and target variable for train and test datasets
                input_feature_train_df = train_df.drop(columns=["Selling Price"])
                target_feature_train_df = train_df[["Selling Price"]]
                input_feature_test_df = test_df.drop(columns=["Selling Price"])
                target_feature_test_df = test_df[["Selling Price"]]
                
                # Apply preprocessing object on training and testing dataframes
                logging.info("Applying preprocessing object on training and testing dataframes.")
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                
                # 1. Transform only the target labels into log values (to stabilize target distribution)
                target_feature_train_arr = np.log1p(np.array(target_feature_train_df))
                target_feature_test_arr = np.log1p(np.array(target_feature_test_df))

                # 2. Combine the processed feature arrays and target arrays horizontally side-by-side
                train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
                test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

                logging.info("Saving preprocessing object.")
                save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )

                return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            except Exception as e :
                raise CustomException(e , sys)
                
        
