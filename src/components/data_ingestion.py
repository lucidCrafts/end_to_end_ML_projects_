
import csv
import sys
sys.path.append(r'C:\DataScience\Visual studio\end_to_end_ML_projects_\Creditcard_fraud_detection_01')
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


from src.utils import df_balancer
from src.utils import save_object


from imblearn.over_sampling import SMOTE


@dataclass

class DataIngestionConfig:
     
     train_data_path: str= os.path.join('artifacts',"train.csv")
     train_set_resampled_path: str= os.path.join('artifacts',"train_resampled.csv")
     test_data_path: str= os.path.join('artifacts',"test.csv")
     val_data_path: str= os.path.join('artifacts',"val.csv")
     raw_data_path: str= os.path.join('artifacts',"data.csv")
     preprocessor_obj_path: str= os.path.join('artifacts',"preprocessor.pkl")


class DataIngestion:
    
     def __init__(self):
          self.ingestion_config = DataIngestionConfig()
     
     def initiate_data_ingestion(self):
          logging.info("Data Ingestion method is runnning..")
          
          try:
              
               os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
               
               
               current_directory = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
               relative_path = 'notebook\local_dataset\creditcard.csv'
               data_file_path_cc = os.path.join(current_directory, relative_path)
                             
               
               df = pd.read_csv(data_file_path_cc)
               logging.info("The local dataset is loaded to a variable")
               
               #df = df_balancer(df)
               logging.info("The dataset is being balanced..")     
                                                  
               df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
               logging.info("The train test split is starting")
               #df.head()
               
               train_set, test_set = train_test_split(df,test_size=0.2, random_state=1,stratify=df['Class'])
               test_set, val_set = train_test_split(test_set, test_size=.5, random_state= 2, stratify=test_set['Class'])
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
               
               #logging.info("Saving the preprocessing object")
               
               logging.info("The data ingested, and is ready to save to the artifacts directory")
               
               return (self.ingestion_config.train_set_resampled_path, self.ingestion_config.test_data_path, self.ingestion_config.val_data_path)

               
          
          except Exception as e:
                   raise CustomException(e,sys)
              


          
          
if __name__=="__main__":
     
     obj =DataIngestion()
     train_set_resampled_path, test_data_path, val_data_path  = obj.initiate_data_ingestion()
          
     data_transformation= DataTransformation()    
     train_arr, test_arr, val_arr, _ =data_transformation.initiate_data_transformation(train_set_resampled_path, test_data_path, val_data_path)   
     
     modeltrainer = ModelTrainer()
     print(modeltrainer.initiate_model_trainer(train_arr, test_arr, val_arr))
     
     