import os
import sys
import dill
import pickle

import os
import numpy as np 
import pandas as pd


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from .logger import logging


def evaluate_models(X_train, y_train, X_test, y_test, X_val, y_val, models, params):
    try:
        report = {}
        
        for model_name, model_instance in models.items():
            # Access the parameters for the current model
            model_params = params.get(model_name, {})  # Get the parameters from the 'params' dictionary
            
            
            logging.info(f"Evaluating model: {model_name}")
            
            gs = GridSearchCV(model_instance, model_params, cv=3)
            gs.fit(X_train, y_train)
            
            model_instance.set_params(**gs.best_params_)
            model_instance.fit(X_train, y_train)
            
            y_train_pred = model_instance.predict(X_train)
            y_val_pred = model_instance.predict(X_val)
            y_val_pred_binary = (y_val_pred > 0.5).astype(int)
            
            precision = precision_score(y_val, y_val_pred_binary, zero_division=0)
            recall = recall_score(y_val, y_val_pred_binary)
            roc_auc = roc_auc_score(y_val, y_val_pred_binary)
            f1 = f1_score(y_val, y_val_pred_binary)

            report[model_name] = {
                'Precision': precision,
                'Recall': recall,
                'ROC-AUC': roc_auc,
                'F1-score': f1,
            }           
                 
        return report
    except Exception as e:
        raise CustomException(e, sys)


        
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_path:
            pickle.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_path:
            return pickle.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)


""" def evaluate_models(X_train,y_train, X_test, y_test, X_val, y_val,models):
    try: 
        report={}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train) 
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
            
            return report
    except Exception as e:
        raise CustomException(e,sys)
    

"""






































def neural_net(X_train, y_train, X_val, y_val):
    
    neural_network = Sequential()
    neural_network.add(InputLayer((X_train.shape[1],)))
    neural_network.add(Dense(5, activation='relu'))
    neural_network.add(BatchNormalization())
    neural_network.add(Dense(1, activation='sigmoid'))

    neural_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('neural_network', save_best_only=True)

    summary = neural_network.summary()
    summary
    num_epochs = 100
    
    neural_network = neural_network.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, callbacks=[checkpoint])
    return summary, neural_network
#Call this Function by # summary, neural_network= neural_net(X_train, y_train, X_val, y_val)  
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def df_balancer(df):
    not_frauds = df.query('Class == 0')
    frauds = df.query('Class == 1')
    not_frauds['Class'].value_counts(), frauds['Class'].value_counts()
    df= df.sample(len(frauds), random_state=33)
    df = df.sample(frac=1, random_state=1)
    return df 



                                                                                                                                
def Robust_Scaler_M(df,coloumns_to_scale):
     
    rc = RobustScaler()
    
    for c in coloumns_to_scale:
      df[c]= rc.fit_transform(df[c].to_numpy().reshape(-1,1))

    return df

#coloumns_to_scale =['Time','Amount']
#Call this fucntion Robust_Scaler_M(df,coloumns_to_scale)


''''

def evaluate_models(X_train, y_train, X_test, y_test, X_val, y_val, models, params):
    try:
        report = {}
        
        for model_name, model in models.items():
            
            
            
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_val_pred_binary = (y_val_pred > 0.5).astype(int)
            
            precision = precision_score(y_val, y_val_pred_binary, zero_division=0)
            recall = recall_score(y_val, y_val_pred_binary)
            roc_auc = roc_auc_score(y_val, y_val_pred_binary)
            f1 = f1_score(y_val, y_val_pred_binary)

            report[model_name] = {
                'Precision': precision,
                'Recall': recall,
                'ROC-AUC': roc_auc,
                'F1-score': f1,  # Include the computed F1-score in the report
            }           
                 
        return report
    except Exception as e:
        raise CustomException(e, sys)


'''