import pandas as pd
import numpy as np
import cobra as cb
from functools import partial

import pickle
import random
import time
import os
import sys
import subprocess
import ast

#for latin hypercube sampling
from scipy.stats import qmc

#for binary PSO problem formulation
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

#for generation of optimization problem
from utils.DesignScores import *

#for the construction of the objective function based on a GEM score
### Function to Generate LHS-aided representation of population ###
def generate_LHS_gcSwarms_combinatorial_space(maximum_deletion_number, target_list, n_particles):
    samples = []
    
    for de in range(1, maximum_deletion_number+1):
        print(f"Generating representation of {de} deletion designs")
        remaining_samples = int(n_particles/maximum_deletion_number)+1 #get same number of samples for each deletion number
        
        while remaining_samples > 0:
            sampler = qmc.LatinHypercube(d=de)
            pop = sampler.integers(l_bounds=0, u_bounds=len(target_list), n=remaining_samples)
            valid_designs = [design for design in pop
                             if len(set(design))== len(list(design)) # As each number represents a design, it does not makes sense to repeat a number in a design
                             and not any([np.array_equal(design, sample) for sample in samples])]
            
            samples += valid_designs            
            remaining_samples -= len(valid_designs)
            
    #transform to binary space
    samples = np.array([ [int(target_list.index(feature) in design) for feature in target_list ]
                          for design in samples ])
    
    return samples[:n_particles] #in the case that generated samples outnumber n_particles, favour the selection of low deletion number
        
        
def binary_to_designs(designs, reaction_list):
    #Transform Binary Output to deletion strategy
    
    return [ '-'.join([reaction for deletion, reaction in zip(design, reaction_list) if deletion==1.0])
             for design in designs
           ]


def pso_result_to_library(optimizer, reaction_list, mode='binary'):
    
    cost_record = [ -c for swarm_costs in optimizer.all_cost_history for c in swarm_costs]
    position_record = [ p  for swarm_positions in optimizer.pos_history for p in swarm_positions]
    generation_record = [gen for gen, swarm_costs in enumerate(optimizer.all_cost_history) for c in swarm_costs]
    
    if mode=='binary':
        
        df = pd.DataFrame([[gen]+list(p)+[c] for gen, p, c in zip(generation_record, position_record, cost_record)],
                          columns = ['Generation']+reaction_list+['Score'])
        
    elif mode=='designs':
        design_record = binary_to_designs(position_record, reaction_list)
        
        df = pd.DataFrame.from_dict({'Generation': generation_record, 'Designs': design_record, 'Score': cost_record})
        
    else:
        print('ERROR: The values enabled for parameter "mode" are "binary" or "designs"')
    
    df = df.drop_duplicates()
    
    return df


def gcSwarms_optimization_function(X, model_setup, target_score, max_deletions):
    '''
    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
    The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
    The computed score for each design
    '''
    contextualized_objective_function = GEM_objective_function(model_setup, target_score, max_deletions)

    return contextualized_objective_function.compute(X)


def gcSwarms(model_setup, optimization_function, max_deletions, combination_number=100000, options={'c1': 2.0,'c2': 2.0,'w': 1.0,'k': float,'p': 1}, init_method='LHS', groups=4, cores=6):
    cost_history = []
    pos_history = []
    n_iterations = 100
    available_methods = ['LHS', 'random', 'pre_computed']
    assert init_method in available_methods, f"{init_method} is not a valid initialization, please use one of {', '.join(available_methods)}"
    
    print('Generating initialization samples through Latin Hypercube Sampling...')
    n_particles = int(combination_number/n_iterations)
    dimensions = len(model_setup['reaction_list'])
    options['k'] = n_particles/groups # fragment the particles in subgroups
    
    # Construct initial population by LHS
    if init_method == 'LHS':
        samples_pos = generate_LHS_gcSwarms_combinatorial_space(max_deletions, model_setup['reaction_list'], n_particles)

    # Construct initial population by random sampling
    elif init_method == 'random':
        samples_pos = np.array([np.array(random.choices((1, 0), weights=[(max_deletions/2)/dimensions, 1-((max_deletions/2)/dimensions)], k=dimensions)) for _ in range(n_particles)])
    
    elif init_method == 'pre_computed':
        print('Not implemented yet!')

    print('Constructing optimization problem...')
    optimizer = BinaryPSO(n_particles=n_particles,
                          dimensions=dimensions,
                          options=options,
                          init_pos=samples_pos)
    
    print('Running gcSwarms search...')
    cost, pos = optimizer.optimize(optimization_function, iters=n_iterations, n_processes=cores)
    
    print('Search ended!')
    print("Best cost:", cost)
    print("Best position:", pos)
    # Display search history
    plot_cost_history(cost_history=optimizer.cost_history)
    
    print('Saving results in a pandas.DataFrame structure')
    results = pso_result_to_library(optimizer, model_setup['reaction_list'], mode='binary')
    results = results[model_setup['reaction_list'] + ["Score"]] # REMOVE THE "GENERATION" COLUMN
    results = results.fillna(0)
    results = results.drop_duplicates()
    
    return results


def get_PSO_top_designs(binary_designs_df, reaction_list, max_kos, score_threshold):
    # The function takes as input the results dataframe that is generated by gcSwarms
    # function and returns the found designs within above a score threshold
    top_binary_designs = []

    print('Filtering Top Designs...') 
    top_binary_designs_df = binary_designs_df.loc[binary_designs_df['Score']>=score_threshold].drop_duplicates()
    
    top_binary_designs = [[int(d) for d in design]
                          for design in top_binary_designs_df[reaction_list].values.tolist()]
    
    print('Parsing binary to KO strategy...')
    top_designs = binary_to_designs(top_binary_designs, reaction_list)
    
    return top_binary_designs,  top_designs
    
