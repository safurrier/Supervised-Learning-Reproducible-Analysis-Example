#!/usr/bin/env python
# coding: utf-8

# In[7]:


import ruamel.yaml as yaml
import os
import sys
import pandas as pd
import numpy as np
import pandas_profiling
import altair as alt
from selenium import webdriver
from itertools import product

alt.themes.enable('opaque')
#%matplotlib inline


NO_CONFIG_ERR_MSG = """No config file found. Root directory is determined by presence of "config.yaml" file."""

original_wd = os.getcwd()

# Number of times to move back in directory
num_retries = 10
for x in range(0, num_retries):
    if os.path.exists("config.yaml"):
        break
    else:
        os.chdir('../')
        # If reached the max number of directory levels change to original wd and print error msg
        if x+1 == num_retries:
            os.chdir(original_wd)
            print(NO_CONFIG_ERR_MSG)


# Add directory to PATH
path = os.getcwd()

if path not in sys.path:
    sys.path.append(path)
    
from analysis_helpers import *    


# ## Params

# In[8]:


RESULTS_PATH = 'reports/output/'

#### MUST HAVE CHROMEDRIVER INSTALLED SOMEWHERE WTIH THIS POINTING TO IT
# Set Driver for Exporting
driver = webdriver.Chrome(r'C:\Users\afurrier\chromedriver.exe')


# In[9]:


data_iterative_alg_combos = list(product(['cars', 'madelon'],  ['ANN', 'Boost', 'SVMLin', 'SVMRBF']))
data_alg_combos = list(product(['cars', 'madelon'],  ['ANN', 'Boost', 'DT', 'KNN', 'SVMLin', 'SVMRBF']))
data_alg_combos


# ## Class to contain necessary Graphs for every algorithm

# In[10]:


iterative_status_df = {
    'ANN':True,
    'Boost':True,
    'DT':False,
    'KNN':False,
    'SVMLin':True,
    'SVMRBF':True
}


# ## Overall Scores

# In[11]:


results = pd.read_csv('reports/output/test-results-reduced-cars.csv')
results['Best_Params'] = results.loc[:, 'Hyperparameters':].astype(str).sum(axis=1)
alt.Chart(results).mark_bar().encode(
    x='Algorithm:N',
    y=alt.Y('Score:Q', scale=alt.Scale(domain=[0.0, 1.0])),
    column='Dataset:N',
    color='Dataset:N'
)


# ## Best number of Iterators per Alg/Dataset

# In[12]:


iterations = pd.read_csv('reports/output/best_number_of_iterations_by_data_algorithm_params_cars.csv')
             #.rename('Number_Of_Iterations':'Best_Number_Of_Iterations'))
iterations


# ## Learning Curves side by side

# In[15]:


data_iterative_alg_combos = list(product(['cars', 'madelon'],  ['ANN', 'Boost', 'SVMLin', 'SVMRBF']))
learners =  ['ANN', 'DT', 'Boost', 'KNN', 'SVMLin', 'SVMRBF']
sets = ['cars', 'madelon']


# ## Export Side by Side Learning Curves and Timing Curves

# Learning Curves

# In[25]:


for algorithm in learners:
    learning_curve_plots = []
    for dataset in sets:
        analysis_plotter = AlgorithmResults(results_fpath=RESULTS_PATH, 
                               dataset=dataset,
                               algorithm=algorithm,
                              iterative=iterative_status_df[algorithm])
        learning_curve_plots.append(analysis_plotter.plot_learning_curve.properties(
    title=f'{dataset}'))
    one, two = learning_curve_plots
    print(algorithm)
    combined_chart = (one | two)
    combined_chart.save(f'reports/figures/learning_curves/dataset_combined/{algorithm}.png', scale_factor=2.0)
    combined_chart.save(f'reports/figures/all/dataset_combined/{algorithm}_learning_curve.png', scale_factor=2.0)    


# Timing Curves

# In[26]:


for algorithm in learners:
    timing_curves = []
    for dataset in sets:
        analysis_plotter = AlgorithmResults(results_fpath=RESULTS_PATH, 
                               dataset=dataset,
                               algorithm=algorithm,
                              iterative=iterative_status_df[algorithm])
        timing_curves.append(analysis_plotter.plot_timing_curve.properties(
    title=f'{dataset}'))
    one, two = timing_curves
    print(algorithm)
    combined_chart = (one | two)
    combined_chart.save(f'reports/figures/timing_curves/dataset_combined/{algorithm}.png', scale_factor=2.0)
    combined_chart.save(f'reports/figures/all/dataset_combined/{algorithm}_timing_curves.png', scale_factor=2.0)    


# In[ ]:


for dataset, algorithm in data_alg_combos:
    test_object = AlgorithmResults(results_fpath=RESULTS_PATH, 
                                   dataset=dataset,
                                   algorithm=algorithm,
                                  iterative=iterative_status_df[algorithm])
    
    print(f'Algorithm: {algorithm}')
    print(f'Dataset: {dataset}')    
    
    learning_curve = test_object.plot_learning_curve
    learning_curve.save(f'reports/figures/learning_curves/{algorithm}_{dataset}.png', scale_factor=2.0)
    learning_curve.save(f'reports/figures/all/{algorithm}_{dataset}_learning_curve.png', scale_factor=2.0) 
    
    timing_curve = test_object.plot_timing_curve
    timing_curve.save(f'reports/figures/timing_curves//{algorithm}_{dataset}.png', scale_factor=2.0)
    timing_curve.save(f'reports/figures/all/{algorithm}_{dataset}_timing_curve.png', scale_factor=2.0)     
    of_curve = test_object.plot_overfitting_curve
    if of_curve:
        of_curve.save(f'reports/figures/of_curves//{algorithm}_{dataset}.png', scale_factor=2.0)
        of_curve.save(f'reports/figures/all/{algorithm}_{dataset}_OF_curve.png', scale_factor=2.0)     
    


# In[ ]:


of_dfs = []
for dataset, algorithm in data_iterative_alg_combos:
    test_object = AlgorithmResults(results_fpath=RESULTS_PATH, 
                                   dataset=dataset,
                                   algorithm=algorithm,iterative=iterative_status_df[algorithm])
    of_dfs.append(test_object.overfitting_curve_dfs)


# In[ ]:


of_dfs_together = pd.concat(of_dfs)
of_dfs_together
alt.Chart(of_dfs_together).mark_line().encode(
    x=alt.X('Max_Iterations/Estimators:Q'),
    y=alt.Y('Score:Q'),
    column='Dataset',
    color='Algorithm',
).transform_filter(
    (alt.datum.Hyperparameters == 'No_Regularization') & (alt.datum.Split == 'Test')
)


# In[ ]:




