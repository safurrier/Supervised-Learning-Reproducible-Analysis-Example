import ruamel.yaml as yaml
import os
import sys
import pandas as pd
import numpy as np
import altair as alt
alt.themes.enable('opaque')


### Class for pulling data and plotting key analysis
class AlgorithmResults(object):
    import altair as alt
    
    def __init__(self, results_fpath=None, algorithm=None, dataset=None, iterative=False):
        assert results_fpath, 'Must specify a directory with results outputs'
        assert algorithm, 'Must specify an algorithm'
        assert dataset, 'Must specify a dataset'
        self.results_fpath = results_fpath
        self.algorithm = algorithm
        self.dataset = dataset
        self.iterative = iterative
        self.df_dict = get_df_dict_from_algorithm_dataset(self.results_fpath,
                                                         algorithm=self.algorithm,
                                                         dataset=self.dataset)
        self.learning_curve_dfs = get_learning_curve_dfs(self.df_dict,
                                                        dataset=self.dataset,
                                                        algorithm=self.algorithm)
        self.timing_curve_dfs = get_timing_curve(self.df_dict,
                                                        dataset=self.dataset,
                                                        algorithm=self.algorithm)
        if self.iterative:
            self.overfitting_curve_dfs = get_overfitting_curve(self.df_dict,
                                                        dataset=self.dataset,
                                                        algorithm=self.algorithm)
        
    @property
    def plot_learning_curve(self, 
                            x=alt.X('Datapoints:Q'), 
                            y=alt.Y('CV_Mean:Q', scale=alt.Scale(domain=[0.0, 1.0])),
                           color=alt.Color('Split:N')):
        """ Plot the learning curve of the algorithm and dataset"""
        return alt.Chart(self.learning_curve_dfs).mark_line().encode(
        x=x,
        y=y,
        color=color
    )
    
    @property
    def plot_timing_curve(self, 
                            x=alt.X('Training_Fractional_Size:Q'), 
                            y=alt.Y('Time:Q'),
                           color=alt.Color('Function:N')):
        """ Plot the learning curve of the algorithm and dataset"""
        return alt.Chart(self.timing_curve_dfs).mark_line().encode(
            x=x,
            y=y,
            color=color,
        )
    
    @property
    def plot_overfitting_curve(self, 
                                x=alt.X('Max_Iterations/Estimators:Q'), 
                                y=alt.Y('Score:Q', scale=alt.Scale(domain=[0.0, 1.0])),
                               color=alt.Color('Split:N'),
                                  column=alt.Column('Hyperparameters:N')):
            """ Plot the learning curve of the algorithm and dataset"""
            # Only runs if learner is iterative
            if self.iterative:
                return alt.Chart(self.overfitting_curve_dfs).mark_line().encode(
                x=x,
                y=y,
                color=color,
                column=column)
            else:
                return None


# Pull Data from output directory helpers

def file_search_and_replace(directory, search, replace, verbose=True):
    """Given a directory, search all filenames in it for the regex
    pattern provided. If found, replace with the provided string 
    by renaming.
    Set verbose=True to see which files are renamed"""
    from pathlib import Path
    import re
    # Make path out of provided directory
    directory_path = Path(directory)
    # Search directory fielnames
    for filename in os.listdir(directory_path):
        # If there's a pattern match
        if re.search(search, filename):
            # Create a new filename replacing the old pattern
            new_fname = re.sub(search, replace, filename)
            # Rename it
            os.rename(directory_path / filename, directory_path / new_fname)
            # If verbose print the renamed files
            if verbose:
                print(f'Rename:\n{directory_path / filename}\nTo:\n{directory_path / new_fname}\n\n')
                
                
def get_df_dict_from_algorithm_dataset(directory_path, dataset=None, 
                                       algorithm=None):
    
    from pathlib import Path
    results_path = Path(directory_path)

    # Loop through and find relevant files
    relevant_files = []
    # Dictionary to hold relevant files
    df_dict = {}
    
    for file in os.listdir(results_path):
        fname_split = file.split("_")
        # Split last string on extension
        file_ext = fname_split.pop()
        fname_split = fname_split + file_ext.split('.')
        #print(fname_split)
        if (dataset in fname_split) & (algorithm in fname_split):
            relevant_files.append(file)
            
    for fname in relevant_files:
        key = fname.split('.')[0]
        df_dict[key] = pd.read_csv(results_path / fname)        
            
    return df_dict

def tidy_learning_curve(base_df, split=None, dataset=None, algorithm=None):
    """Convert a learning curve df into tidy format depending
    on train/test split specified"""
    assert split.upper() in ['TRAIN', 'TEST'], \
    'Split must be "Train" or "Test"'
    CV_cols = ['CV_1', 'CV_2', 'CV_3', 'CV_4', 'CV_5']
    tidy_df = base_df.copy()
    tidy_df.columns = ['Data_Points'] + CV_cols
    tidy_df['CV_Mean'] = tidy_df[CV_cols].mean(axis=1)
    tidy_df['Split'] = split
    tidy_df[['Data_Points', 'Split', 'CV_Mean']]
    
    # Optionally add algorithm and dataset
    if algorithm:
        tidy_df['Algorithm'] = algorithm
    if dataset:
        tidy_df['Dataset'] = dataset    
    return tidy_df

def get_learning_curve_dfs(df_dict, algorithm=None, dataset=None):
    """Given a dictionary with keys of fnames and values
    of dataframes, search for learning curve dfs, tidy them 
    and return them as a single dataframe"""
    learning_curve_dfs = []
    # Search in df_dict for LC which indicates
    # Learning curve df
    for key, df in df_dict.items():
        if 'LC' in key.split('_'):
            #print(key)
            split = key.split("_")[-1]
            #print(f'Split: {split}')
            #display(df.head())
            # Tidy data and append to list
            learning_curve_dfs.append(tidy_learning_curve(df, split=split,
                                                         algorithm=algorithm,
                                                         dataset=dataset))
    # Concat together
    #[display(df) for df in learning_curve_dfs]
    learning_curve_df = pd.concat(learning_curve_dfs)
    return learning_curve_df
    
def tidy_timing_curve(timing_df, algorithm=None, dataset=None):
    """Convert a learning curve df into tidy format"""

    tidy_timing_df = timing_df.copy()
    # Add first column name
    #tidy_timing_df.iloc[:, 0].name = 'Training_Fractional_Size'
    tidy_timing_df = tidy_timing_df.rename(columns={tidy_timing_df.columns[0]: 'Training_Fractional_Size'})
    cols = tidy_timing_df.columns.values.tolist()
    # Capitalize
    tidy_timing_df.columns = [col[0].upper() + col[1:] for col in cols]
    #tidy_timing_df.columns = ['Training_Fractional_Size', 'Train', 'Test']
    # Originally the first column is the test size
    # So setting the training size is just 1 - this
    tidy_timing_df['Training_Fractional_Size'] = (1 - tidy_timing_df['Training_Fractional_Size']).round(2)
    tidy_timing_df = tidy_timing_df.melt(id_vars='Training_Fractional_Size', var_name='Function', value_name='Time')
    tidy_timing_df['Function'] = tidy_timing_df['Function'].map({'Test':'Predict', 'Train':'Fit'})
    
    # Optionally add algorithm and dataset
    if algorithm:
        tidy_timing_df['Algorithm'] = algorithm
    if dataset:
        tidy_timing_df['Dataset'] = dataset
    return tidy_timing_df
    
def get_timing_curve(df_dict, algorithm=None, dataset=None):
    timing_curves = []
    for key, df in df_dict.items():
        if 'timing' in key.split('_'):
            timing_df = df.copy()
            timing_df = tidy_timing_curve(timing_df, algorithm=algorithm, dataset=dataset)
            timing_curves.append(timing_df)
    timing_curve_df = pd.concat(timing_curves)
    return timing_curve_df
    
def tidy_iter_df(iter_df, param_set=None, dataset=None, algorithm=None):
    # Melt data
    # Find max iteration columns
    iteration_column = iter_df.filter(regex='(_max_iter$|_estimators$|_iter$)').columns.values.tolist()[0]
    
    tidied_iter_df = (pd.melt(iter_df, id_vars=iteration_column, var_name='Split', value_name='Score')
                     .rename(columns={iteration_column:'Max_Iterations/Estimators'}))
    # Clean up split
    tidied_iter_df.Split = tidied_iter_df.Split.str.split(' ').str.get(0).str.title()
    # Add apram_set (e.g. overfitting or not), dataset, algorithm if applicacable
    tidied_iter_df['Hyperparameters'] = param_set    
    if algorithm:
        tidied_iter_df['Algorithm'] = algorithm
    if dataset:
        tidied_iter_df['Dataset'] = dataset
    return tidied_iter_df
    
def get_overfitting_curve(df_dict, algorithm=None, dataset=None):
    iter_dfs = []
    for key, df in df_dict.items():
        if 'ITERtestSET' in key.split('_'):
            if 'OF' in key.split('_'):
                of_iter_df = df.copy()
                of_iter_df = tidy_iter_df(of_iter_df, param_set='No_Regularization', 
                                          algorithm=algorithm, dataset=dataset)
                iter_dfs.append(of_iter_df)
            else:
                non_of_iter_df = df.copy()
                non_of_iter_df = tidy_iter_df(non_of_iter_df, param_set='Best_Hyperparameters', 
                                          algorithm=algorithm, dataset=dataset)
                iter_dfs.append(non_of_iter_df)
    iter_dfs = pd.concat(iter_dfs)
    return iter_dfs
    
def get_optimal_iteration_number(df_dict, params='best'):
    """Pull the optimal number of iterations to run an 
    algorithm given either the 'best' hyperparams or 
    'overfitting' params"""
    assert params in ['best', 'overfitting'], \
    "params arg must be either 'best' or 'overfitting'"
    iterations_df_dict = {}
    for key, df in df_dict.items():
        # Check if it's the search across iteration 
        # number
        if key.startswith('ITER_base'):
            # Add to relevant dfs
            iterations_df_dict[key] = df
    
    # Set best iteration equal to None 
    # to begin with
    
    OF_best_iteration = None
    best_iteration = None
    for key, df in iterations_df_dict.items():
        key_splits = key.split('_')
        # Check if it's an overfitting df
        if 'OF' in key_splits:
            # Colum with iteration numbers is one that ends with 'max_iter'
            iter_column = df.filter(regex='(_max_iter$|_estimators$|_iter$)').columns.values.tolist()
            assert iter_column, """No matching colum that ends with "_max_iter" or "n_estimators" 
            or "n_iter" found in file {}""".format(key)        
            # Sort by best test score and the locate the iterations column
            OF_best_iteration = df.sort_values(by='rank_test_score').loc[:, iter_column].values[0][0]
        # Otherwise it's the best hyperparameters search
        else:
            # Colum with iteration numbers is one that ends with 'max_iter'
            iter_column = df.filter(regex='(_max_iter$|_estimators$|_iter$)').columns.values.tolist()
            #print(df)
            assert iter_column, """No matching colum that ends with "_max_iter" or "n_estimators" 
            or "n_iter" found in file {}""".format(key)           
            # Sort by best test score and the locate the iterations column
            best_iteration = df.sort_values(by='rank_test_score').loc[:, iter_column].values[0][0]
    
    if params=='best':
        if best_iteration:
            return best_iteration
        else:
            return np.nan
    elif params=='overfitting':
        # If no best iteration was able to be pulled, 
        # return a NaN
        if OF_best_iteration:
            return OF_best_iteration
        else:
            return np.nan
        
def get_all_best_iters(fpath, datasets=None, algorithms=None, tidy=True):
    from itertools import product
    dataset_algorithm_pairs = list(product(datasets, algorithms))
    
    rows = []
    for dataset_algorithm_combo in dataset_algorithm_pairs:
        df_dict = get_df_dict_from_algorithm_dataset(fpath, 
                                                     dataset=dataset_algorithm_combo[0],
                                                     algorithm=dataset_algorithm_combo[1])
        overfitting_best_iteration = get_optimal_iteration_number(df_dict, params='overfitting')
        best_iteration = get_optimal_iteration_number(df_dict, params='best')
        rows.append([dataset_algorithm_combo[0], dataset_algorithm_combo[1], 
                    best_iteration, overfitting_best_iteration])
    best_iterations = pd.DataFrame(rows, columns=['Dataset', 'Algorithm', 
                                                'Best_Params_Iterations', 'No_Regularization_Params'])
    if tidy:
        best_iterations = pd.melt(best_iterations, id_vars=['Dataset', 'Algorithm'],
                                 var_name='Hyperparameters', value_name='Number_Of_Iterations')
        best_iterations['Hyperparameters'] = (best_iterations['Hyperparameters'].str.split('_')
                                             .str.slice(0, -1)
                                             .str.join('_')
                                            )
    return best_iterations
   
            