import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
     AdaBoostRegressor,
     GradientBoostingRegressor,
     RandomForestRegressor,
)

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models
from src.utils import neural_net
from src.utils import load_object



@dataclass
class ModelTrainerConfig:
     trained_model_file_path = os.path.join('artifacts',"model.pkl")


class ModelTrainer:
     def __init__(self):
          self.model_trainer_config=ModelTrainerConfig()
          
     def initiate_model_trainer(self, train_array, test_array, val_array): # Add ", preprocessor_path" as another argument if you want 
          try:
               logging.info("Split training and test input data")
               X_train, y_train,X_test,y_test, X_val, y_val =(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1],
                    val_array[:,:-1],
                    val_array[:,-1],
               )
               
               logging.info(print(X_train.shape, y_train.shape,X_test.shape,y_test.shape, X_val.shape, y_val.shape))
               
               
               models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Logistic Regression": LogisticRegression(),
                # Add more models here if needed
               }
               
               params = { 
               "Decision Tree": { 
               #'criterion': ['mse', 'friedman_mse', 'mae'],  # Example criterion values
               #'max_depth': [None, 5, 10, 20],  # Example max_depth values
               # Add more DecisionTreeRegressor parameters here
               },
               "Gradient Boosting": { 
               #'n_estimators': [50, 100, 200],  # Example n_estimators values
               #'learning_rate': [0.01, 0.1, 0.2],  # Example learning_rate values
               # Add more GradientBoostingRegressor parameters here
               },
               "K-Neighbors": {
               #'n_neighbors': [3, 5, 10],  # Example n_neighbors values
               #'weights': ['uniform', 'distance'],  # Example weights values
               # Add more KNeighborsRegressor parameters here
               },
               "XGBoost": { 
               #'n_estimators': [50, 100, 200],  # Example n_estimators values
               #'learning_rate': [0.01, 0.1, 0.2],  # Example learning_rate values
               # Add more XGBRegressor parameters here
               },
               "AdaBoost": {  
               #'n_estimators': [50, 100, 200],  # Example n_estimators values
               #'learning_rate': [0.01, 0.1, 0.2],  # Example learning_rate values
               # Add more AdaBoostRegressor parameters here
               },
               "Logistic Regression": { 
               #'C': [0.1, 1.0, 10.0],  # Example C values
               #'penalty': ['l1', 'l2'],  # Example penalty values
               # Add more LogisticRegression parameters here
               },
          # Add more models and their parameters here if needed
               }

               
               model_report:dict = evaluate_models(X_train =X_train, y_train=y_train, X_test= X_test, y_test=y_test, X_val=X_val, y_val=y_val, models=models,params=params)
               
               
               best_model_score = 0.5
                                                           
                    
                               

                            
               if best_model_score<0.5:
                    raise CustomException("No best model found")
               
               logging.info("EMPTY, its supposed find a best model here")
               
               #preprocessing_obj = load_objects(file_path)
               
               
               
               
               #save_object(file_path = ModelTrainerConfig.trained_model_file_path,
                 #          obj=best_model                     )
              
               
               return  model_report
                     
               
          except Exception as e:
               raise CustomException(e, sys)
               