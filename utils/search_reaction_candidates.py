import sys
import os
import pickle
import cobra as cb

from designFunctions import * 

from joblib.externals.loky import get_reusable_executor

get_reusable_executor().shutdown(wait=True)

print('Starting search of reaction candidate to delete...')
config = pickle.load(open('search_candidates_config.pkl', 'rb'))

blocked_reactions =  set([r.id for r in config['model'].reactions])-set(get_rxn_with_fva_flux(config['model'], fraction_of_objective=0.9, loopless=True))
targets = get_KO_candidate_list(config['model'],
                                config['essential_rxns'],
                                blocked_reactions=blocked_reactions)

pickle.dump(targets, open(config['targets_output'], 'wb'))
print('Search ended!')
