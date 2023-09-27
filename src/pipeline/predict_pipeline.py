import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass





class predict_pipelineConfig:
     model_path = os.path.join('artifacts',"model.pkl")
     preprocessor_path = os.path.join('artifacts',"preprocessor.pkl")



class PredictPipeline:
     def __init__(self):
          
          pass
     
     def predict(self, features):
          
          try:
               
          
               model = load_object(file_path=predict_pipelineConfig.model_path)
               preprocessor = load_object(file_path=predict_pipelineConfig.preprocessor_path)
               data_scaled = preprocessor.transform(features)
               predictions = model.predict(data_scaled)
               return predictions
          
          except Exception as e:
               raise CustomException(e, sys)
          
          




class CustomData:
     def __init__( self,
                  Time,
                  V1 : float,
                  V2: float,
                  V3: float,
                  V4: float,
                  V5: float,
                  V6: float,
                  V7: float,
                  V8: float,
                  V9: float,
                  V10: float,
                  V11: float,
                  V12: float,
                  V13: float,
                  V14: float,
                  V15: float,
                  V16: float,
                  V17: float,
                  V18: float,
                  V19: float,
                  V20: float,
                  V21: float,
                  V22: float,
                  V23: float,
                  V24: float,
                  V25: float,
                  V26: float,
                  V27: float,
                  V28: float,
                  Amount: float,
                  Class :int
               ):
          
               self.V1=V1   
                  
               self.V2=V2   
                  
               self.V3=V3   
                  
               self.V4=V4   
                  
               self.V5=V5   
                  
               self.V6=V6   
                  
               self.V7=V7   
                  
               self.V8=V8   
                  
               self.V9=V9   
                  
               self.V10=V10   
                  
               self.V11=V11   
                  
               self.V12=V12   
                  
               self.V13=V13   
                  
               self.V14=V14   
                  
               self.V15=V15   
                  
               self.V16=V16   
                  
               self.V17=V17   
                  
               self.V18=V18   
                  
               self.V19=V19   
                  
               self.V20=V20   
                  
               self.V21=V21   
                  
               self.V22=V22   
                  
               self.V23=V23   
                  
               self.V24=V24   
                  
               self.V25=V25   
                  
               self.V26=V26   
                  
               self.V27=V27   
                  
               self.V28=V28   
                  
               self.Amount=Amount   
                  
               self.Class=Class
               
     def get_data_as_frame(self):
          
          try:
               custom_data_input_dict = {
                    "V1 ":[self.V],
                    "V2":[self.V],
                    "V3":[self.V],
                    "V4":[self.V],
                    "V5":[self.V],
                    "V6":[self.V],
                    "V7":[self.V],
                    "V8":[self.V],
                    "V9":[self.V],
                    "V10":[self.V],
                    "V11":[self.V],
                    "V12":[self.V],
                    "V13":[self.V],
                    "V14":[self.V],
                    "V15":[self.V],
                    "V16":[self.V],
                    "V17":[self.V],
                    "V18":[self.V],
                    "V19":[self.V],
                    "V20":[self.V],
                    "V21":[self.V],
                    "V22":[self.V],
                    "V23":[self.V],
                    "V24":[self.V],
                    "V25":[self.V],
                    "V26":[self.V],
                    "V27":[self.V],
                    "V28":[self.V],
                    "Amount":[self.Amount],
                    "Class": [self.Class],   
                    
               }
               return pd.DataFrame(custom_data_input_dict)
     
          except Exception as e:
               raise CustomException(e, sys)