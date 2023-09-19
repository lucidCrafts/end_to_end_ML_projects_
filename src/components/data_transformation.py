import os
import sys
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder,StandardScaler

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
from src.utils import load_object



@dataclass
class DataTransformationConfig:
    
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor_obj.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["Time","Amount"]
          
            num_pipeline= Pipeline(
                steps=[
                ("scaler",RobustScaler())

                ])
          

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor_obj=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                                ]
            )
            
                              
            return preprocessor_obj
        
        
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path, val_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            val_df =pd.read_csv(val_path)
            
            preprocessing_obj =self.get_data_transformer_object()
               
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            columns_to_scale = ["Time", "Amount"]

            input_feature_train_df=train_df.drop(columns="Class",axis=1)
            target_feature_train_df=train_df["Class"]

            input_feature_test_df=test_df.drop(columns="Class",axis=1)
            target_feature_test_df=test_df["Class"]
            
            input_feature_val_df = val_df.drop(columns="Class",axis=1)
            target_feature_val_df = val_df["Class"]

            

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            input_feature_val_arr=preprocessing_obj.transform(input_feature_val_df)
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            
                
                    
            logging.info(f"Saved preprocessing object.")

            preprocessing_obj = save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj


            )

            return (
                train_arr,
                test_arr,
                val_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)