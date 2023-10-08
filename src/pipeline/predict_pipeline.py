import sys
import os
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object
from src.config import PipelineConfig
from tensorflow import keras

class PredictPipeline:
    def __init__(self):
        self.PredictPipelineConfig_filepath = PipelineConfig()

    def predict(self, features):
        try:
            print(self.PredictPipelineConfig_filepath.trained_model_file_path)
            #print("C:\DataScience\Visual studio\end_to_end_ML_projects_\Creditcard_fraud_detection_01\src\components\artifacts\model.h5")
            #model = load_object(file_path=self.PredictPipelineConfig_filepath.trained_model_file_path)
            model = keras.models.load_model(self.PredictPipelineConfig_filepath.trained_model_file_path)
            #tensor_model_nobugs = keras.models.load_model(self.PredictPipelineConfig_filepath.best_model_file_path)
            
            preprocessor = load_object(file_path=self.PredictPipelineConfig_filepath.preprocessor_obj_path)
            
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
 
        except Exception as e:
            print(e)
            
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 Time: float,
                 V1: float, V2: float, V3: float, V4: float, V5: float,
                 V6: float, V7: float, V8: float, V9: float,
                 V10: float, V11: float, V12: float, V13: float, V14: float,
                 V15: float, V16: float, V17: float, V18: float, V19: float,
                 V20: float, V21: float, V22: float, V23: float, V24: float,
                 V25: float, V26: float, V27: float, V28: float,
                 Amount: float, Class: int):

        self.Time = Time
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.V5 = V5
        self.V6 = V6
        self.V7 = V7
        self.V8 = V8
        self.V9 = V9
        self.V10 = V10
        self.V11 = V11
        self.V12 = V12
        self.V13 = V13
        self.V14 = V14
        self.V15 = V15
        self.V16 = V16
        self.V17 = V17
        self.V18 = V18
        self.V19 = V19
        self.V20 = V20
        self.V21 = V21
        self.V22 = V22
        self.V23 = V23
        self.V24 = V24
        self.V25 = V25
        self.V26 = V26
        self.V27 = V27
        self.V28 = V28
        self.Amount = Amount
        self.Class = Class

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "Time": [self.Time],
                "V1": [self.V1],
                "V2": [self.V2],
                "V3": [self.V3],
                "V4": [self.V4],
                "V5": [self.V5],
                "V6": [self.V6],
                "V7": [self.V7],
                "V8": [self.V8],
                "V9": [self.V9],
                "V10":[self.V10],
                "V11":[self.V11],
                "V12":[self.V12],
                "V13":[self.V13],
                "V14":[self.V14],
                "V15":[self.V15],
                "V16":[self.V16],
                "V17":[self.V17],
                "V18":[self.V18],
                "V19":[self.V19],
                "V20":[self.V20],
                "V21":[self.V21],
                "V22":[self.V22],
                "V23":[self.V23],
                "V24":[self.V24],
                "V25":[self.V25],
                "V26":[self.V26],
                "V27":[self.V27],
                "V28": [self.V28],
                "Amount": [self.Amount],
                #"Class": [self.Class],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ =="__main__":
    #print(PipelineConfig().trained_model_file_path)
    #model = keras.models.load_model(PipelineConfig().trained_model_file_path)
    #print(PipelineConfig().val_data_path)
    
    #val_data = pd.read_csv(PipelineConfig().val_data_path)
    #X_val = val_data.drop(columns='Class')  # Assuming 'Class' is your target column

    
    #single_instance = X_val.iloc[[1], :]
    
    #predictions = model.predict(single_instance)
    #print(predictions)
    
    

    # Instantiate the PredictPipeline
    ppppipeline_ = PredictPipeline()
    
    # Load sample data for testing
    val_data = pd.read_csv(PipelineConfig().val_data_path)
    X_val = val_data.drop(columns='Class')  # Assuming 'Class' is your target column
    single_instance = X_val.iloc[[1], :]
    
    # Get predictions
    predictions = ppppipeline_.predict(single_instance)
    
    # Print results
    print(f"Predicted value: {predictions}")
