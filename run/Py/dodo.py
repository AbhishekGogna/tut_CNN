#!/usr/bin/env python3
# load functions -------------------------------------------------------------------------
import sys
source_code_at = '/proj/src'
sys.path.append(source_code_at)
from Py.libs import *
#from Py.func 

print("works")

# Define input_files ----------------------------------------------------------------------------------------------------------
res_at = "/proj/inputs"
input_paths = {}
input_paths['g_data_a'] = f'{res_at}/gdata_add.feather'
input_paths['p_data'] = f'{res_at}/blues_acr_env.feather'