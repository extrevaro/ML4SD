import sys
import os
import pickle
import re
from typing import Optional

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.metrics

import autosklearn.regression
from ConfigSpace.configuration_space import ConfigurationSpace
import autosklearn.pipeline.components.data_preprocessing
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT

#As data will be pre-processed as a previous step, define a class
class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()  # Return an empty configuration as there is None


# Add NoPreprocessing component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)

replicate, data_dir = sys.argv[1:]

if __name__ == "__main__":                         #Needed when using parallel execution of Auto-Sklearn
	os.environ['OPENBLAS_NUM_THREADS'] = '1'   #Safer to execute parallel runs setting this value according to developers
	print('Loading dataframe...', flush=True)
	chunks = pd.read_csv(data_dir, chunksize=10**6)
	input_df = pd.concat(chunks)

	def bp_to_components(bp, end_tag):
	'''
	Handle function to transform a bioprocess tag extracted from file
	to a sustrate-product pair
	'''
	    bp = bp.replace('_EX', '&EX')

	    sustrate = bp.split('&')[0]
	    product = bp.split('&')[1].replace(end_tag, '')

	    return [sustrate, product]


	print('Performing pre-processing of dataframe...', flush=True)
	#Instead of bioprocess as one variable, we will create variables Sustrate and Product, reducing
	#the combinatorial design space (there are several shared metabolites in bioprocesses) and enabling
	#to learn potential relations between metabolites and patterns in designs
	end_tag = '_chemical_space_ko_results'
	unique_bioprocesses = len(input_df['Bioprocess'].unique())
	bps = np.array([bp_to_components(bp, end_tag) for bp in input_df['Bioprocess']], dtype='object') #Creates a list of lists with this pattern [[Sustrate, Product], ..., [Sustrate, Product]]
	input_df['Sustrate'], input_df['Product'] = zip(*bps)
	input_df.drop(columns=['Bioprocess'], inplace=True)
	# Identify columns to one-hot encode
	columns_to_encode = ['Sustrate', 'Product']
	# Use pd.get_dummies() to one-hot encode specified columns
	input_df = pd.get_dummies(input_df, columns=columns_to_encode)
	# shuffle the DataFrame rows to avoid bias in design selection for training
	input_df = input_df.sample(frac = 1)

	print('Setting the features, objective and learning setup...', flush=True)
	target_var = 'yield'
	feature_list = list(set(input_df.columns)-{target_var})
	#Define training time
	train_time = 20*24*3600 #20 days
	# Define the features and target
	X = input_df[feature_list].values
	y = input_df[target_var].values
	# Split the data into training and testing sets stratified by multiple columns
	X_train, X_test, y_train, y_test = train_test_split(
	    X,
	    y,
	    test_size=0.30,
	    random_state=8
	)

	del input_df, X, y

	print('Training an Ensemble regressor model during %u seconds...' % int(train_time), flush=True)
	#Build and fit a regressor
	automl = autosklearn.regression.AutoSklearnRegressor(
	    time_left_for_this_task=train_time,
	    per_run_time_limit=int(train_time/10),
	    resampling_strategy="cv",
	    max_models_on_disc=10,
	    include = {
		       "regressor": ["gradient_boosting", "random_forest", "mlp", "decision_tree"], #models found in specialist models for restricting the searchspace
		       "data_preprocessor": ["NoPreprocessing"] #data is already preprocessed so delete this option for restricting searchspace
		       },
	    resampling_strategy_arguments= {
		                            "train_size": 0.8,      # The size of the training set
		                            "shuffle": True,        # Whether to shuffle before splitting data
		                            "folds": 5              # Used in 'cv' based resampling strategies
		                        },
        dataset_compression = {
                                "memory_allocation": 0.2,   #related to the parameter 'memory_limit'
                                "methods": ["precision", "subsample"]
                            },
	    n_jobs=3,
	    memory_limit=1536000, #150 Gb
            #tmp_folder="/data/sbg/strain-design/autosklearn_classification_example_tmp", NOTE: I have modified ~/autosklearn/automl_common/common/utils/backend.py to write tmp file on data folder
            delete_tmp_folder_after_terminate =False                                     #      to handle 'No space left on device' error when tmp dir is out of memory (Only happens with big models)
	)

	automl.fit(X_train, y_train, dataset_name="strain-design")
	automl.refit(X_train, y_train)
	print('Model trained!')

	# save model as a pickle file
	model_pkl_file = f"/data/strain-design/results/generalist_regressor_model_{replicate}.pkl"

	result = { 'train_data' : [X_train, X_test, y_train, y_test], 'model' : automl}

	with open(model_pkl_file, 'wb') as file:  
	    pickle.dump(result, file)
