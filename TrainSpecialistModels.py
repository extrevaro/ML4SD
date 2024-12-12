import sys
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
import autosklearn.regression
from sklearn import metrics

project_list = [('EX_4hbz_e', 'EX_fum_e'), ('EX_4hbz_e', 'EX_3hpp(e)'), ('EX_4hbz_e', 'EX_6ax_c'), ('EX_4hbz_e', 'EX_adpt(e)'), ('EX_4hbz_e', 'EX_etoh_e'), ('EX_acald_e', 'EX_adpt(e)'), ('EX_acald_e', 'EX_etoh_e'), ('EX_acald_e', 'EX_fum_e'), ('EX_acald_e', 'EX_succ_e'), ('EX_cit_e', 'EX_3hpp(e)'), ('EX_cit_e', 'EX_adpt(e)'), ('EX_cit_e', 'EX_akg_e'), ('EX_cit_e', 'EX_fum_e'), ('EX_cit_e', 'EX_mal__L_e'), ('EX_cit_e', 'EX_pyr_e'), ('EX_cit_e', 'EX_succ_e'), ('EX_glc__D_e', 'EX_3hpp(e)'), ('EX_glc__D_e', 'EX_fum_e'), ('EX_glc__D_e', 'EX_sbt_D(e)'), ('EX_glc__D_e', 'EX_succ_e'), ('EX_glu__L_e', 'EX_6ax_c'), ('EX_glu__L_e', 'EX_adpt(e)'), ('EX_glu__L_e', 'EX_akg_e'), ('EX_glu__L_e', 'EX_etoh_e'), ('EX_glu__L_e', 'EX_pyr_e'), ('EX_oaa_e', 'EX_3hpp(e)'), ('EX_oaa_e', 'EX_adpt(e)'), ('EX_oaa_e', 'EX_pyr_e'), ('EX_oaa_e', 'EX_succ_e'), ('EX_octa_e', 'EX_14bdo(e)'), ('EX_octa_e', 'EX_3hpp(e)'), ('EX_octa_e', 'EX_adpt(e)'), ('EX_octa_e', 'EX_etoh_e'), ('EX_octa_e', 'EX_pyr_e'), ('EX_octa_e', 'EX_succ_e'), ('EX_phe__L_e', 'EX_3hpp(e)'), ('EX_phe__L_e', 'EX_6ax_c'), ('EX_phe__L_e', 'EX_adpt(e)'), ('EX_phe__L_e', 'EX_akg_e'), ('EX_phe__L_e', 'EX_fum_e'), ('EX_phe__L_e', 'EX_mal__L_e'), ('EX_phe__L_e', 'EX_pyr_e'), ('EX_phe__L_e', 'EX_succ_e'), ('EX_pyr_e', 'EX_3hpp(e)'), ('EX_pyr_e', 'EX_6ax_c'), ('EX_pyr_e', 'EX_adpt(e)'), ('EX_pyr_e', 'EX_fum_e'), ('EX_skm_e', 'EX_3hpp(e)'), ('EX_skm_e', 'EX_adpt(e)'), ('EX_skm_e', 'EX_mal__L_e'), ('EX_tol_e', 'EX_14bdo(e)'), ('EX_tol_e', 'EX_3hbl(e)'), ('EX_tol_e', 'EX_3hpp(e)'), ('EX_tol_e', 'EX_etoh_e'), ('EX_tol_e', 'EX_fum_e'), ('EX_tol_e', 'EX_succ_e'), ('EX_uri_e', 'EX_14bdo(e)'), ('EX_uri_e', 'EX_3hpp(e)'), ('EX_uri_e', 'EX_6ax_c'), ('EX_uri_e', 'EX_adpt(e)'), ('EX_uri_e', 'EX_etoh_e'), ('EX_uri_e', 'EX_fum_e'), ('EX_uri_e', 'EX_sbt_D(e)'), ('EX_uri_e', 'EX_succ_e')]

C = 1 #DBTL Cycle number
target_var = 'yield'
train_time = 120*60 #2h; selected by iterating through a range of values and computing validation scores (see file time_selection.csv within project folder); for gcfront is 600 (10min)

strain_design_algorithm = "gcswarms"
project_tag = 'chemical_space_ko_results'
root_project_dir = 'chemical_space/PSO4SD'
library_filename = 'gcSwarm_binary_design_library.csv'

'''
times_to_test = [20*60, 40*60, 60*60, 120*60, 240*60]

model_comparation_data = {'Time' : [],
                          'R2_Train' : [],
                          'R2_Test' : [],
                          'MAE' : [],
                          'MRAE' : [],
                          'MSE' : []}
'''

for bp in project_list:
#for train_time in times_to_test:
	#construct paths for interest files
	project = '_'.join(list(bp)+[project_tag])
	dirpath = '/'.join([root_project_dir, project])
	library_filename = '/'.join([dirpath, library_filename])
	input_df = pd.read_csv(library_filename).drop(columns=['Unnamed: 0'])

	if strain_design_algorithm == 'gcfront':
		#compute yield of the design to compare with gcSwarms
		model = cb.io.load_matlab_model('/'.join([dirpath, 'configured_model.mat']))
		target_carbons = [m for m in model.reactions.get_by_id(bp[1]).metabolites][0].elements['C']
		uptake_carbons = [m for m in model.reactions.get_by_id(bp[0]).metabolites][0].elements['C']
		uptake = model.reactions.get_by_id(bp[0]).bounds[0]
		input_df[target_var] = (input_df['ProductFlux']*target_carbons)/(abs(uptake)*uptake_carbons)
		
		#transform data to binary dataset able to train ensemble model
		reaction_list = set([ r for design in input_df['ReactionDeletions'].tolist() for r in design.split(" ")])
		binary_designs = [ [int(r in design) for r in reaction_list]+[target] for design, target in zip(input_df['ReactionDeletions'].tolist(), input_df[target_var])]
		transformed_df = pd.DataFrame(binary_designs, columns=list(reaction_list)+[target_var])
	
	if strain_design_algorithm == 'gcswarms':
		reaction_list = list(set(input_df.columns.tolist()[:-1])-set([target_var]))

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
	
	print('Training an Ensemble regressor model for project %s during %u seconds...' % (project, train_time))
	#Build and fit a regressor
	automl = autosklearn.regression.AutoSklearnRegressor(
	    time_left_for_this_task=train_time,
	    resampling_strategy="cv",                               #Cross-Validation
	    resampling_strategy_arguments= {
                                            "train_size": 0.8,      # The size of the training set
                                            "shuffle": True,        # Whether to shuffle before splitting data
                                            "folds": 5              # Used in 'cv' based resampling strategies
                                        },

	    memory_limit=10240,                                     #In Mb
	)

	automl.fit(X_train, y_train, dataset_name="strain-design")
	automl.refit(X_train.copy(), y_train.copy())               #When using cv in Auto-Sklearn,it is needed to use this method to train the ensemble with the entire dataset
	print('Model trained!')

	# save the iris classification model as a pickle file
	model_pkl_file = '/'.join([dirpath, f"regressor_model_{C}.pkl"])   

	result = { 'train_data' : [X_train, X_test, y_train, y_test], 'model' : automl}

	with open(model_pkl_file, 'wb') as file:  
	    pickle.dump(result, file)
	'''
	train_predictions = automl.predict(X_train)
	test_predictions = automl.predict(X_test)
	#Compute r2 and errors
	model_comparation_data['Time'].append(train_time)
	model_comparation_data['R2_Train'].append(metrics.r2_score(y_train, train_predictions))
	model_comparation_data['R2_Test'].append(metrics.r2_score(y_test, test_predictions))
	model_comparation_data['MAE'].append(metrics.mean_absolute_error(y_test, test_predictions))
	model_comparation_data['MRAE'].append(metrics.mean_absolute_percentage_error(y_test, test_predictions))
	model_comparation_data['MSE'].append(metrics.mean_squared_error(y_test, test_predictions))
	'''
	
#model_comparation_df = pd.DataFrame.from_dict(model_comparation_data)
#model_comparation_df.to_csv('/'.join([dirpath, "time_selection.csv"]))

