#!/usr/bin/env python3
# load functions -------------------------------------------------------------------------
import sys
source_code_at = '/proj/src'
sys.path.append(source_code_at)
from Py.libs import *
from Py.func import *

proj_paths = read_json("/proj/inputs/core_paths.json")

# Define input_files ----------------------------------------------------------------------------------------------------------
res_at = "/proj/inputs"
input_paths = {}

## data for predictions
input_paths['acr_g.npy'] = f'{res_at}/acr_g.npy'
input_paths['acr_p.csv'] = f'{res_at}/acr_p.csv'
input_paths['acr_g.scl'] = f'{res_at}/acr_g.scl'
input_paths['acr_p.scl'] = f'{res_at}/acr_p.scl'

## cv scenarios
input_paths['acr_cv'] = f'{res_at}/acr_cv.json'

## Dump input files
if not os.path.exists(res_at):
    os.makedirs(res_at, exist_ok = True)
write_json(input_paths, f'{res_at}/input_paths.json')

# Define tasks ----------------------------------------------------------------------------------------------------------------
def task_create_slurm_scripts():
    '''creates slurm scripts'''
    input_file_path = f'{proj_paths["inputs"]}/input_paths.json'
    
    task_name = 'create_slurm_scripts'
    tag_name = f'{proj_paths["run_Py"]}/{task_name}'
    output_path = f'{proj_paths["outputs"]}/{task_name}'
    target_file = f'{proj_paths["outputs"]}/{task_name}_res.json'  
    return {
        'file_dep': [input_file_path, f'{tag_name}.py'],
        'targets': [target_file],
        'actions': [f'python3 {tag_name}.py {source_code_at} {input_file_path} {output_path} {task_name} {target_file} > {tag_name}.log 2> {tag_name}.err'],
    }