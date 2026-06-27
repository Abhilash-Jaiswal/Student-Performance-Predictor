import os 
import sys # helps to get the exception details
import pandas as pd
import numpy as np
from src.exceptions import CustomException
import dill   # helps to save the object in the file path


def save_object(file_path, obj):
    '''This function is used to save the object in the specified file path using dill library.'''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

