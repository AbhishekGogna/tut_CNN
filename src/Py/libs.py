# Add common libraries here to be loaded for all analysis

## Python build-in
import os, sys, importlib, inspect, string
import time, random, gc, pickle, json, logging, re, math

from os.path import exists
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #each ID always refers to the same GPU card
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #each process only sees the appropriate GPU card(s)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # to supress the tensorflow warning messages for compiliation
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#tf.test.is_built_with_cuda() # check if tf was built with cuda 
#tf.config.list_physical_devices('GPU') # check if gpus are detected

# Data wrangling
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 10)
import pyarrow.feather as ft
import numpy as np
import patsy as pt
import matplotlib.pyplot as plt
#import pydot
from pprint import pformat
from datetime import datetime