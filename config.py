import pandas as pd
import json
import os

# Global Variables - Dont Delete any of this irrespective of Used/Unused
ts_col_name = []
column_names_dropdown = []
col_name_map_dict = dict() # Not in use
problem_type = ''
filtered_dataset = pd.DataFrame() # Not in use
target_column = ''
best_model=''
reduced_batches = []
