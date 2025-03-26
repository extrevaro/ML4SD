import ast
import re
import os
import sys
import time
import random

#import multiprocessing as mp
import joblib
import pandas as pd
import numpy as np

from cameo.flux_analysis.simulation import pfba
from cameo.flux_analysis.analysis import phenotypic_phase_plane
from functools import partial

#DESIGN ANALYSIS
from cameo.visualization.plotting.with_plotly import PlotlyPlotter
from cameo.strain_design.deterministic.flux_variability_based import FSEOF
import cobra


def get_rxn_with_fva_flux(model, fraction_of_objective=0.2, loopless=False):
    from cobra.flux_analysis import flux_variability_analysis
    model_tmp = model.copy()
    #get fva results for all reactions in the model
    print('Performing fva...')
    fva_result = flux_variability_analysis(model_tmp, fraction_of_optimum=fraction_of_objective,  loopless=loopless)
    del model_tmp
    
    print('Getting reactions from fva')
    #Filter out those reaction tath have min and max fluxes equals to 0:
    rxn_with_flux = fva_result.loc[(fva_result['maximum'] != 0) | (fva_result['minimum'] != 0) ].index.tolist()

    return rxn_with_flux

def custom_minimal_media(model, carbon_sources):
    from cobra.medium import minimal_medium
    max_growth = model.slim_optimize()
    minimal_media = minimal_medium(model, max_growth)
    custom_media = { r: (minimal_media[r], model.reactions.get_by_id(r).upper_bound) for r in minimal_media.index }
    for cs in carbon_sources.keys():
        custom_media[cs] = (carbon_sources[cs], model.reactions.get_by_id(cs).upper_bound)
    
    model.medium = {r: abs(custom_media[r][0]) for r in custom_media.keys()}
    
    return model, custom_media


def display_KO_candidates_results(model, ko_targets, target_biomass, target_reaction):
    for s in ko_targets:
        print('Results for deletion of %s' % ', '.join([r for r in s]))
        with model:
            for r in s:
                model.reactions.get_by_id(r).knock_out()

            plotter = PlotlyPlotter()
            pfba_solution = pfba(model)
            mutant_biomass = pfba_solution.fluxes[target_biomass]
            mutant_target = pfba_solution.fluxes[target_reaction]
            print('Flux of biomass solution is %s and the flux of target reaction is %s' % (mutant_biomass, mutant_target))
            try:
                result = phenotypic_phase_plane(model,
                                                variables=[model.reactions.get_by_id(target_biomass)],
                                                objective=model.reactions.get_by_id(target_reaction))
                result.plot(plotter)
            except:
                print('Infeasible solution, trying_next')
                continue

            for r in s:
                display(model.reactions.get_by_id(r))


#this function will create the protected reactions.
#This set consists on those reactions whose deletion will seriously hamper the flux
#of both target biomass and metabolite exchange
def select_protected_reactions(model, target_biomass, target_reaction, reactions_to_test=None, flux_fraction=0.1):
    wt_sol = model.optimize()
    growth_threshold = wt_sol.objective_value*flux_fraction
    minimum_production = wt_sol.fluxes[target_reaction]*flux_fraction
    protected_reactions = []

    if reactions_to_test:
        deletion_list = [r for r in model.reactions if r.id in reactions_to_test]
    else:
        deletion_list = model.reactions

    print('Checking %s reactions...' % len(deletion_list))
    for r in deletion_list:
        test_model = model.copy()
        model.objective = target_biomass
        test_model.reactions.get_by_id(r.id).knock_out()
        try:
            pfba_result = pfba(test_model)
            no_flux_biomass = pfba_result[target_biomass] < growth_threshold
            no_flux_metabolite = pfba_result[target_reaction] < minimum_production

        except Exception as e:
            print(e)
            print(r.id)
            protected_reactions.append(r.id)
            continue

        if no_flux_biomass or no_flux_metabolite:
            print(r.id)
            protected_reactions.append(r.id)

    return set(protected_reactions)

    
def check_reaction_essentiality_in_parallel(model_settings, r):
    is_protected_reaction = False

    test_model = model_settings['model']
    #ensure model has the same constraints as in the definition of model_settings
    for rxn, bounds in model_settings['model_bounds'].items():
        test_model.reactions.get_by_id(rxn).bounds = bounds
        
    test_model.objective = model_settings['target_biomass']

    test_model.reactions.get_by_id(r.id).bounds = (0,0)
        
    try:
        pfba_result = pfba(test_model)
        no_flux_biomass = pfba_result[model_settings['target_biomass']] < model_settings['growth_threshold']
        no_flux_metabolite = pfba_result[model_settings['target_reaction']] < model_settings['minimum_production']
        
        if no_flux_biomass or no_flux_metabolite:
            is_protected_reaction = True

    except Exception as e:
        print(e)
        print(r.id)
        is_protected_reaction = True
        
    if is_protected_reaction:
    	return r.id

       

def select_protected_reactions_in_parallel(model, target_biomass, target_reaction, reactions_to_test=None, flux_fraction=0.1):
    wt_sol = model.optimize()

    if reactions_to_test:
        deletion_list = [r for r in model.reactions if r.id in reactions_to_test]
        
    else:
        deletion_list = model.reactions
    
    
    model_settings = { 'model' : model.copy(),
    			'model_bounds' : {r.id : r.bounds for r in model.reactions},
                       'target_biomass' : target_biomass,
                       'target_reaction' : target_reaction,
                       'growth_threshold' : wt_sol.objective_value*flux_fraction,
                       'minimum_production' : wt_sol.fluxes[target_reaction]*flux_fraction
                     }

    #parallelize work
    print('Parallelizing task..')
    processes = 4
    check_model = partial(check_reaction_essentiality_in_parallel, model_settings)    
    result_list = joblib.Parallel(backend="loky", n_jobs=processes, batch_size=round(len(deletion_list)/processes), verbose=10)(joblib.delayed(check_model)(r) for r in deletion_list)
                
    print('End of parallel task!')
    
    #delete empty responses from check_model
    
    return [r for r in result_list if not r is None]


def delete_trasport_reactions(model, rxn_list):
    filtered_list=[]
    for r in rxn_list:
        sustrate_compartiments = {met.id[:-2] : met.id[-2:] for met in model.reactions.get_by_id(r).reactants}
        product_compartiments = {met.id[:-2] : met.id[-2:] for met in model.reactions.get_by_id(r).products}
        shared_mets = set(sustrate_compartiments.keys()) & set(product_compartiments.keys())
        if all([sustrate_compartiments[met] == product_compartiments[met] for met in shared_mets]):
            filtered_list.append(r)
    print(len(filtered_list))
    return filtered_list


def find_coupled_reactions(model, reaction_to_test):
    """Find reaction sets that are structurally forced to carry equal flux"""
    model = model.copy()
    stoichiometries = {}
    for reaction in reaction_to_test:
        for met, coef in model.reactions.get_by_id(reaction).metabolites.items():
            stoichiometries.setdefault(met.id, {})[reaction] = coef

    # Find reaction pairs that are constrained to carry equal flux
    couples = []
    for met_id, stoichiometry in stoichiometries.items():
        if len(stoichiometry) == 2 and set(stoichiometry.values()) == {1, -1}:
            couples.append(set(stoichiometry.keys()))

    # Aggregate the pairs into groups
    coupled_groups = []
    for couple in couples:
        for group in coupled_groups:
            if len(couple & group) != 0:
                group.update(couple)
                break
        else:
            coupled_groups.append(couple)

    return coupled_groups

#function for generate target list of knocks outs.
#Uses a model and a reaction set of protected reactions computed by the function
#select_protected_reactions. For coherent results when used with no blocked_reactions
#(default), it should be called with the same epsilon and tolerance used to generate
#the consistent model through fastcc
def get_KO_candidate_list(model, protected_reactions, blocked_reactions=None, carbon_limit=16, epsilon=1e-4, tolerance=1e-7):
    #exclude protected_reactions:
    candidate_set = set([r.id for r in model.reactions]) - set(protected_reactions)
    #exclude blocked reactions
    if blocked_reactions is None:
        blocked_reactions = Fastcore.fast_find_blocked(model, epsilon=epsilon, tolerance=tolerance)
    candidate_set = candidate_set - blocked_reactions
    #exclude non GPR, exchange and spontaneous reactions
 
    candidate_list = [ r for r in candidate_set if
                      model.reactions.get_by_id(r).gene_reaction_rule != '' and #Non GPR
                      len(model.reactions.get_by_id(r).metabolites) > 1 and     #Exchange
                      'spontaneous' not in model.reactions.get_by_id(r).name ]  #Spontaneous 
    
    #exclude transport reactions
    candidate_list = delete_trasport_reactions(model, candidate_list)
    #exclude reactions using high carbon molecules (>16 C atoms)
    
    cofactor_list = ['atp', 'adp', 'coa', 'nadph', 'nadp', 'nadh', 'nad', 'fad', 'fadh2']
    high_carbon_reactions=[]
    for r in candidate_list:
        for subs in model.reactions.get_by_id(r).reactants:
            formula = subs.formula
            id = subs.id
            if not any([cofactor in id for cofactor in cofactor_list]):
                try:
                    n_of_carbons = int(re.findall(r'C([0-9]{1,2})', formula)[0])
                except:
                    continue
                if n_of_carbons > carbon_limit:
                    high_carbon_reactions.append(r)
                    break
    

    candidate_list = list(set(candidate_list)-set(high_carbon_reactions))
    

    #delete reactions that are coupled with others in the list
    
    rxns_groups = find_coupled_reactions(model, candidate_list)
    candidate_list = [ r for r in candidate_list if
                       r not in [not_leader for rxn_group in rxns_groups for not_leader in list(rxn_group)[1:]] ]

    return candidate_list



#To avoid the incorrect filtering of blocked reactions due to
#diverging fluxes towards non-objective biomass reactions, a
#function will be defined which purge all biomass but the target
#one from the model. For this, we are going to asume that biomas  
#reactions meet thefollowing criteria:
#    1. They have non integer coefficients for reaction reactants
#    2. They are the top reactions concerning the nº of metabolites
#After deleting all biomass reactions, the function will check
#if the model is feasible and if it is not, it will raise a 
#warning and return the given model
def purge_non_objective_biomass(model, target_biomass, n_of_biomass_reactions=None, threshold=12):
    new_model = model.copy()
    target_biomass_rxn = new_model.reactions.get_by_id(target_biomass)
    reactions_with_nonint_coeffs = [r for r in new_model.reactions 
                                    if all([not coeff.is_integer() for coeff in r.metabolites.values() if coeff<0])]
    
    
    n_of_metabolites_dict = { r.id : len(r.metabolites) for r in reactions_with_nonint_coeffs }
    n_of_metabolites_dict = dict(sorted(n_of_metabolites_dict.items(), key=lambda item: item[1], reverse=True))

    
    if n_of_biomass_reactions is None:
        #if no number of biomass reactions is given, filtering of reactions will be
        #done by a defined thershold
        n_of_biomass_reactions = threshold
        
    biomass_reactions = list(n_of_metabolites_dict.keys())[:n_of_biomass_reactions]
            
    for r in biomass_reactions:
        if r != target_biomass :
            #assure the reaction is not an intermediary biomass reaction supporting
            #the BOF with reactants
            rxn_products = [p.id for p in new_model.reactions.get_by_id(r).products]
            if not any([s.id in rxn_products for s in target_biomass_rxn.reactants]):
                new_model.reactions.get_by_id(r).bounds = (0,0)
            
    if new_model.optimize().status == 'optimal':
    
        return new_model
    
    else:
        print('WARNING: THE REACTION PURGING YIELDS AN INFEASIBLE MODEL, PLEASE PURGE THEM MANUALLY')
        print('RETURNING INPUT MODEL...')
        return model
  

def add_missing_reactions(model, repository_model, carbon_source, product):
    
    try:
        repository_model.objective = {repository_model.reactions.get_by_id(product) : 1}
        fba = repository_model.optimize()
                
        reactions_to_synthetize_product = [r.id for r in repository_model.reactions if fba.fluxes[r.id]!=0]          
        reactions_to_add = set(reactions_to_synthetize_product)-set([r.id for r in model.reactions])
        
        if len(reactions_to_add) == 0:
            edited_model = model
            feasible = 0
            print('Repository model have no extra reactions to aid in the bioprocess!')
            
        else:
            edited_model = model.copy()
            edited_model.add_reactions([r for r in repository_model.reactions if r.id in reactions_to_add])
            edited_model.objective = {edited_model.reactions.get_by_id(product) : 1}
            feasible = edited_model.slim_optimize() > 0
            
    except Exception as e:
        print(e)
        edited_model = model
        feasible = False
        print('Metabolite %s, used as a product it is not present in repository model or cannot be produced!' % product)
        
    return edited_model, feasible

