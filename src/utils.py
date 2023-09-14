import os
import sys
import dill
import pickle

import numpy as np 
import pandas as pd


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)



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
      df[c]= rc.fit_transform(new_df[c].to_numpy().reshape(-1,1))

    return df

#coloumns_to_scale =['Time','Amount']
#Call this fucntion Robust_Scaler_M(df,coloumns_to_scale)