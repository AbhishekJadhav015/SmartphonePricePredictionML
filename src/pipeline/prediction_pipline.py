import sys
import os 
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("before loading ")
            model = load_obj(file_path = model_path)
            preprocessor = load_obj(file_path = preprocessor_path)
            print("after loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, Brand , Model , Color , Rating , Original_Price ,Memory , Storage ):
        self.Brand = Brand
        self.Model = Model
        self.Color = Color
        self.Rating = Rating
        self.Original_Price = Original_Price
        self.Memory = Memory
        self.Storage = Storage
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Brand" :[self.Brand],
                "Model" : [self.Model],
                "Color" : [self.Color],
                "Rating" : [self.Rating],
                "Original_Price" : [self.Original_Price],
                "Memory" : [self.Memory],
                "Storage" : [self.Storage]
                
                }
        
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e ,sys)