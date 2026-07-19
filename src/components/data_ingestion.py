import os
import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Configuration class for data ingestion
@dataclass
class DataIngestionConfig :
    train_data_path: str =os.path.join("artifacts","train.csv")
    test_data_path: str =os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","data.csv")

# DataIngestion class for handling data ingestion process 
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    # Method to initiate data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion mehtod or components")
        try :
            # Read the datasets from the specified paths
            data1 = pd.read_csv("C:\\Users\\abhis\\project\\ML_project_with_MLFlow\\notebooks\\data\\Flipkart_mobile_brands_scraped_data.csv")
            data2 = pd.read_csv("C:\\Users\\abhis\\project\\ML_project_with_MLFlow\\notebooks\\data\\Flipkart_Mobiles.csv")
            df = pd.concat([data1,data2],ignore_index=True)
            
            logging.info("Read the datasets as DataFrames")
            
            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path , index=False , header =True)
            logging.info("Train test split initiated")
            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header= True)
            
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e :
            raise CustomException(e,sys)
        

