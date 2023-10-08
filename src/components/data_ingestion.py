
import csv
import sys

import os
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.logger import logging
from src.exception import CustomException

from src.utils import df_balancer
from src.utils import save_object

from pathlib import Path

from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
from src.config import PipelineConfig



class DataIngestion:
    
     def __init__(self):
          self.ingestion_config = PipelineConfig()
     
     def initiate_data_ingestion(self):
          logging.info("Data Ingestion method is runnning..")
          
          try:
              
               self.ingestion_config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
               data_file_path_cc = self.ingestion_config.current_directory_Pathlib / "Creditcard_fraud_detection_01" / "notebook" / "local_dataset" / "creditcard.csv"

               df = pd.read_csv(data_file_path_cc)
               logging.info("The local dataset is loaded to a variable")
               logging.info("The dataset is being balanced..")     
               df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
               logging.info("The train test split is starting")
               train_set, test_set = train_test_split(df, test_size=0.2, random_state=1, stratify=df['Class'])
               test_set, val_set = train_test_split(test_set, test_size=.5, random_state=2, stratify=test_set['Class'])

               logging.info(train_set.head(20))
               
               # Applying SMOTE on training data
               smote = SMOTE(random_state=42)
               X_train = train_set.drop('Class', axis=1)  # assuming 'Class' is your target variable
               y_train = train_set['Class']
               X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

               # Merge features and target variable back into a DataFrame
               train_set_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
                         
                             
               train_set_resampled.to_csv(self.ingestion_config.train_set_resampled_path, index=False, header=True)
               #train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
               test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
               val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
               
               logging.info(f"X_train_resampled: {X_train_resampled.head(5)}  y_train_resampled : {y_train_resampled.head(5)}")
     
               logging.info("The data ingested, and is ready to save to the artifacts directory")
               
               return (self.ingestion_config.train_set_resampled_path, self.ingestion_config.test_data_path, self.ingestion_config.val_data_path ) # add X_train,train_set_resampled for debugging model 

               
          
          except Exception as e:
                   raise CustomException(e,sys)
              


          
          
if __name__=="__main__":
      
     obj =DataIngestion()
     train_set_resampled_path, test_data_path, val_data_path  = obj.initiate_data_ingestion()
          
     data_transformation= DataTransformation()    
     train_arr, test_arr, val_arr =data_transformation.initiate_data_transformation(train_set_resampled_path, test_data_path, val_data_path)   # , _ 
     
     modeltrainer = ModelTrainer()
     print(modeltrainer.initiate_model_trainer(train_arr, test_arr, val_arr))
     
     '''
     ingestion_config = DataIngestionConfig()
     #print(f"Current diectory is : {ingestion_config.current_directory}, and base directory is :{ingestion_config.base_dir}")
     
     #train_set_resampled_path, test_data_path, val_data_path, X_train,train_set_resampled = DataIngestion().initiate_data_ingestion()
     #print(X_train.head(5))
     #print(train_set_resampled.head(5))
     
     #obj =DataIngestion()
     #train_set_resampled_path, test_data_path, val_data_path  = obj.initiate_data_ingestion()
          
     #data_transformation= DataTransformation()    
     #train_arr, test_arr, val_arr, _ =data_transformation.initiate_data_transformation(train_set_resampled_path, test_data_path, val_data_path)   
     print(f" ingestion_config current_directory_Pathlib: {ingestion_config.current_directory_Pathlib}")
     # print(f" ingestion_config  current_directory :{ingestion_config.current_directory}")
     print(f" ingestion_base_dir :{ingestion_config.base_dir}") 
     print(f" ingestion_base_dir_pAth :{ingestion_config.base_dir_pAth}")
     
     
     ''' 