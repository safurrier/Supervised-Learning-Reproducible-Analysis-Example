ml7641-assignment-one
==============================

Assignment for supervised learning ML 7641 Fall 2018 assignment 1
Author: Sean (Alex) Furrier

To create the necesary conda env, run 

> . ./setup_env.sh

To run the pipeline cleaning and processing data before performing 5 
fold cross validated grid search on the 6 algorithms and 2 datasets,
learning curves, timing curves and expressiveness curves run:

> make analysis

Warning: the above is computationally expensive and can take a great deal
of time. 

To run an individual algorithm, run:

> make {algorithm name}

Cross check the makefile for exact commands. 

To generate the plots used in the analysis, run:

> make plots

Submission analysis is in top level folder in file 'sfurrier3-analysis.pdf'

Project Organization
------------

    ├── sfurrier3-analysis.pdf           <- Project Analysis Submisson
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original data. Mostly contains datasets from UCI
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         description, e.g. `01.0-initial-data-exploration`.
    │
    ├── references         <- Assignment Info.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in analysis
    │   └── output         <- All model assessment outputs, including hyper parameter cv scores
    |                         learning curves, timing curves, expressiveness curves, best 
    |                         number of iterations, final cv scores, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
	|
    ├── setup_env.sh           <- Script to initiailize git repo, setup a conda virtual environment  
    │                         and install dependencies.
    │
    ├── src                <- Source code of various utilies for EDA, cleaning, analyzing etc. 
    │   ├── __init__.py    <- Makes src a Python module
    │   └── utils          <- Utility code for various purposes and packages
    │                 
    ├── helpers.py         <- Utility code for various purposes and packages
    ├── analysis_helpers.py<- Utility code for various analysis (mostly plotting)
    ├── ANN.py             <- Script to run ANN model fitting, param search, learning/timing curve, etc
    ├── Decision_Tree.py   <- Script to run Decision Tree model fitting, param search, learning/timing curve, etc
    ├── Boosting.py        <- Script to run Boosted DT model fitting, param search, learning/timing curve, etc
    ├── KNN.py             <- Script to run KNN model fitting, param search, learning/timing curve, etc
    ├── SVM.py             <- Script to run SVM (linear and RBF kernels) model fitting, param search, learning/timing curve, etc



--------
Model training and evaluation done using pandas, numpy, and scikit-learn.
Visualizations done with altair. 
Code for model training and model evaluation adapted from Jonathan Tay (https://github.com/JonathanTay) 

