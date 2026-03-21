import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #use to create a pipeline for handling both numerical and categorical data
from sklearn.impute import SimpleImputer #use to handle missing values
from sklearn.pipeline import Pipeline #use to create a pipeline
from sklearn.preprocessing import OneHotEncoder #use to handle categorical data
from sklearn.preprocessing import StandardScaler #use to handle numerical data

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is responsible for data transformation based on the different types of data
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), #handles missing values, removing outliers using median strategy
                    ("scaler", StandardScaler()) #doing standard scaling
                ] #this is to be done on training data set and need to be transformed on test data

            )
                
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), #handles missing values in categorical columns # strategy most frequent means replacing missing values with mode
                    ("one_hot_encoder", OneHotEncoder()), #converts categorical variables into dummy/indicator variables
                    ("scaler",StandardScaler(with_mean=False)) #scaling the data without centering it to zero mean, because one hot encoding will create sparse matrix and centering it will make it dense which will increase memory usage
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #for combining both numerical and categorical pipeline, combination of both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns), #pipeline name, what pipeline it is, our columns
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path): #starting my data transformation inside this function
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[

                input_feature_train_arr, np.array(target_feature_train_df)

            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)