import subprocess
import pandas as pd
import numpy as np


def design_to_ml_readable(designs, feature_ordered_list):
    
    return pd.DataFrame(
        [[ int(feature in design) for feature in feature_ordered_list ] for design in designs],
        columns=feature_ordered_list
    )

#print all bioprocesses executed in gcfront
pso_bp_out = subprocess.run("ls -lt PSO4SD/EX_*/gcSwarm_design_library.csv", shell=True, capture_output=True)
pso_bp_fileinfo = pso_bp_out.stdout.decode("utf-8").split("\n")[:-1]
print(f"gcSwarms was executed for {len(pso_bp_fileinfo)} different bioprocesses!")
    
executed_pso = [directory.split('/')[1] for directory in pso_bp_fileinfo]
min_yield = 0.01
pso_bp_status = {}
translated_designs_data = []
bioprocess_tag_data = []


for bp in executed_pso:
    if bp!='EX_glu__L_e_EX_etoh_e_chemical_space_ko_results': #excluded because it had a high number of reactions only in this bioprocess
        pso_solution_path = f"PSO4SD/{bp}/gcSwarm_design_library.csv"
        df = pd.read_csv(pso_solution_path)
        df.dropna(inplace=True)
        design_space = set([r for sd in df['KO_Design'] for r in sd.split('-') ])
        pso_bp_status[bp] = {'max_yield' : df['yield'].max(), 'design_space' : design_space, 'data' : df}
        
        if pso_bp_status[bp]['max_yield']>=min_yield:
            translated_designs_data += pso_bp_status[bp]['data']['KO_Design'].tolist()
            bioprocess_tag_data += [bp]*len(df)            
            #I need to execute DesignScorer here for computing the true values (see in Method_Comparation.iypnb)
            
del df, design_space
    
#print all biprocesses in which pso4sd has not found a solution
pso_solution = [bp for bp, res in pso_bp_status.items() if res['max_yield']>=min_yield]
general_design_space = set.union(*[res['design_space'] for bp, res in pso_bp_status.items() if res['max_yield']>=min_yield])
print(f"gcSwarms found solutions for {len(pso_solution)} different bioprocesses!")
print(f"Design space consists on a total of {len(general_design_space)} reactions!")

translated_designs_data = design_to_ml_readable(translated_designs_data, list(general_design_space))
translated_designs_data['Bioprocess'] = bioprocess_tag_data
translated_designs_data.to_csv('unified_bps.csv')