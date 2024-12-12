import sys
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
import autosklearn.regression

dir_name, project, C, train_time, target_var = sys.argv[1:]
library_filename = '/'.join([dir_name, project, 'binary_design_library.csv'])
input_df = pd.read_csv(library_filename)
reaction_list = input_df.columns.tolist()[:-1]

# Define the features and target
X = input_df[reaction_list]
y = input_df[target_var]

# Split the data into training and testing sets stratified by multiple columns
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=8
)

print('Training an Ensemble regressor model during %u seconds...' % int(train_time))

#Build and fit a regressor
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=int(train_time),
    resampling_strategy="cv",
    resampling_strategy_arguments= {
                                    "train_size": 0.8,      # The size of the training set
                                    "shuffle": True,        # Whether to shuffle before splitting data
                                    "folds": 5              # Used in 'cv' based resampling strategies
                                },
    memory_limit=10240,
    tmp_folder="/tmp/autosklearn_regression_example_tmp",
)

automl.fit(X_train, y_train, dataset_name="strain-design")
automl.refit(X_train, y_train)

print('Model trained!')

# save the iris classification model as a pickle file
model_pkl_file = '/'.join([dir_name, project, 'regressor_model_%s.pkl' % C])   

result = { 'train_data' : [X_train, X_test, y_train, y_test], 'model' : automl}

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(result, file)
