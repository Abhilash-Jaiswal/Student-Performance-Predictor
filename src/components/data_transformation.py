import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer       # for pipeline
from sklearn.impute import SimpleImputer                # for missing value
from sklearn.pipeline import Pipeline               # 
from sklearn.preprocessing import OneHotEncoder,StandardScaler      # transformation , scaling

from src.exceptions import CustomException  
from src.logger import logging
import os               # dealing with directory

from src.utils import save_object       #for saving pickle file

@dataclass  # dataclass is used to create a class that is mainly used to store data and automatically generates special methods like __init__() and __repr__() for the class.
class DataTransformationConfig:
    '''give any path that is required to save the preprocessor object'''
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") 
    # function helps to join the path of the file and directory 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        # this line creates an instance of the DataTransformationConfig class and assigns it 
        # to the data_transformation_config attribute of the DataTransformation class. 
        # This allows the DataTransformation class to access the configuration settings
        # defined in the DataTransformationConfig class.


    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            # this pipeline is used to handle the numerical columns. It first imputes missing 
            # values using the median strategy and then scales the features using StandardScaler.


            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            # this pipeline is used to handle the categorical columns. It first imputes missing
            # values using the most frequent strategy, then applies one-hot encoding to convert 
            # categorical variables into numerical format, and finally scales the features using StandardScaler.


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
# The ColumnTransformer is used to apply different preprocessing pipelines to different subsets of features.

            ''' preprocessor object is created using the ColumnTransformer class, 
            which allows us to apply different preprocessing pipelines to different subsets of features.
            The preprocessor object is then returned from the get_data_transformer_object method,'''
           
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    # we got train and test data path from the data ingestion component 
    def initiate_data_transformation(self,train_path,test_path):

        ''' This function is responsible for initiating the data transformation process. 
        It takes the paths of the train and test datasets as input, reads the data, 
        applies preprocessing transformations, and returns the transformed data along with 
        the path to the saved preprocessor object. '''

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            # this line calls the get_data_transformer_object method to obtain the preprocessing object, 
            # which is responsible for handling the data transformation process.


            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]


            # all data ( train + test ) has to be transformed using the same preprocessor object, 
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info( f"Applying preprocessing object on training dataframe and testing dataframe."  )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            # fit_transform is used on the training data to learn the parameters of the preprocessing steps, 
            # while transform is used on the test data to apply the same transformations learned from the training data.
            #  otherwise, if we use fit_transform on the test data, it would learn new parameters based on the test data, 
            # which could lead to data leakage and biased results.


            train_arr = np.c_[  input_feature_train_arr, np.array(target_feature_train_df) ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # np.c_ is used to concatenate the transformed input features and the target variable into a single array 

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            # The save_object function is called to save the preprocessing object to the specified file path.

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,  # this is path of the saved preprocessor object, which can be used later for inference or deployment of the model.
            )
        except Exception as e:
            raise CustomException(e,sys)
