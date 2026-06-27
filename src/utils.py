import os 
import sys # helps to get the exception details
import pandas as pd
import numpy as np
from src.exceptions import CustomException
import dill   # helps to save the object in the file path
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    '''This function is used to save the object in the specified file path using dill library.'''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models): # ,param
    try:
        report = {} 


        for i in range(len(list(models))):
            model = list(models.values())[i]  # this line is used to get the model object from the models dictionary based on the index i
           
            # para=param[list(models.keys())[i]]
            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)
            # model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # prediction

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            # calculate r2 score for train data and test data

            report[list(models.keys())[i]] = test_model_score
            # store the test model score in the report dictionary

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)
