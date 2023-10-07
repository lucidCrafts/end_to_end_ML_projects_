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
from src.config import PipelineConfig

from pathlib import Path


PREPROCESSOR_OBJ_FILE_PATH=PipelineConfig().preprocessor_path


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import sys
import logging

# Assuming these are defined somewhere:
from src.exception import CustomException
from src.config import PipelineConfig
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = PipelineConfig()

    def get_data_transformer_object(self):
        """
        Returns a data transformer object
        """
        try:
            # Scaling only "Time" and "Amount"
            numerical_columns = ["Time", "Amount"]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", RobustScaler())
                ]
            )

            # V1-V28 don't require any transformation, but need to be retained
            feature_columns = [
                "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
            ]

            # Building the Column Transformer
            preprocessor_obj = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("pass_through", 'passthrough', feature_columns)
                ],
                remainder='drop'  # Droping other columns in the dataframe
            )

            return preprocessor_obj
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, val_path):
        """
        Loads data, transforms it, and saves the transformation object
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)

            preprocessing_obj = self.get_data_transformer_object()

            # Separating input and target features
            X_train = train_df.drop(columns="Class")
            y_train = train_df["Class"]

            X_test = test_df.drop(columns="Class")
            y_test = test_df["Class"]

            X_val = val_df.drop(columns="Class")
            y_val = val_df["Class"]

            # Transforming datasets
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            X_val_transformed = preprocessing_obj.transform(X_val)

            # Merginging transformed features and target
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]
            val_arr = np.c_[X_val_transformed, y_val.to_numpy()]

            # Saving the preprocessing object for later use
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, val_arr

        except Exception as e:
            raise CustomException(e, sys)
