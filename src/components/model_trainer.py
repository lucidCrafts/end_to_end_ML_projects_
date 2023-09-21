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

from src.utils import load_object

from keras.models import Sequential
from keras.layers import InputLayer, Dense, BatchNormalization
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from src.utils import neural_net





from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier



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
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Neural Net": KerasClassifier(build_fn=neural_net, verbose=0),
                # Add more models here if needed
               }
               
               params = { 
               "Decision Tree": {
               'criterion': ['gini', 'entropy'],
               'max_depth': [None, 5, 10, 20],
          },
               "Gradient Boosting": {
               'n_estimators': [50, 100, 200],
               'learning_rate': [0.01, 0.1, 0.2],
          },
               "K-Neighbors": {
               'n_neighbors': [3, 5, 10],
               'weights': ['uniform', 'distance'],
          },
               "XGBoost": {
               'n_estimators': [50, 100, 200],
               'learning_rate': [0.01, 0.1, 0.2],
          },
               "AdaBoost": {
               'n_estimators': [50, 100, 200],
               'learning_rate': [0.01, 0.1, 0.2],
          },
               "Logistic Regression": {
               'C': [0.1, 1.0, 10.0],
               'penalty': ['l1', 'l2', 'elasticnet', 'none'],
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          },
               "Neural Net": {
               'epochs': [5],
               'batch_size': [10, 20],
               'optimizer': ['Adam'],#, 'SGD'
               'neurons': [3],
               'activation': ['relu'],#, 'tanh'
               'input_shape': [(X_train.shape[1],)]
          },
          # Add more models and their parameters here if needed
               }

               
               model_report:dict = evaluate_models(X_train =X_train, y_train=y_train, X_test= X_test, y_test=y_test, X_val=X_val, y_val=y_val, models=models,params=params)
               
               
               best_model_score = 0.5
                                                           
                    
                               

                            
               if best_model_score<0.5:
                    raise CustomException("No best model found")
               
               logging.info("Model training is done..")
               
               #preprocessing_obj = load_objects(file_path)
               
               
               
               
               #save_object(file_path = ModelTrainerConfig.trained_model_file_path,
                 #          obj=best_model                     )
              
               
               return  model_report
                     
               
          except Exception as e:
               raise CustomException(e, sys)
               