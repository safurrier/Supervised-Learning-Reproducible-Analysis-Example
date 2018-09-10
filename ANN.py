# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)



def main():

    abalone = pd.read_hdf('data/processed/datasets.hdf','abalone')        
    abaloneX = abalone.drop('Class',1).copy().values
    abaloneY = abalone['Class'].copy().values

    madelon = pd.read_hdf('data/processed/datasets.hdf','madelon')        
    madelonX = madelon.drop('Class',1).copy().values
    madelonY = madelon['Class'].copy().values



    abalone_trgX, abalone_tstX, abalone_trgY, abalone_tstY = ms.train_test_split(abaloneX, abaloneY, test_size=0.3, random_state=0,stratify=abaloneY)     
    madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     

    pipeA = Pipeline([('Scale',StandardScaler()),
                     ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                     ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                     ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                     ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

    d = abaloneX.shape[1]
    hiddens_abalone = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
    alphasM = [10**-x for x in np.arange(-1,9.01,1/2)]
    d = madelonX.shape[1]
    d = d//(2**4)
    hiddens_madelon = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    params_abalone = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_abalone}
    params_madelon = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_madelon}
    #
    madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params_madelon,'ANN','madelon')        
    abalone_clf = basicResults(pipeA,abalone_trgX,abalone_trgY,abalone_tstX,abalone_tstY,params_abalone,'ANN','abalone')        


    #madelon_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}
    #abalone_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}

    madelon_final_params = madelon_clf.best_params_
    abalone_final_params =abalone_clf.best_params_
    abalone_OF_params =abalone_final_params.copy()
    abalone_OF_params['MLP__alpha'] = 0
    madelon_OF_params =madelon_final_params.copy()
    madelon_OF_params['MLP__alpha'] = 0

    #raise

    #
    pipeM.set_params(**madelon_final_params)  
    pipeM.set_params(**{'MLP__early_stopping':False})                   
    makeTimingCurve(madelonX,madelonY,pipeM,'ANN','madelon')
    pipeA.set_params(**abalone_final_params)
    pipeA.set_params(**{'MLP__early_stopping':False})                  
    makeTimingCurve(abaloneX,abaloneY,pipeA,'ANN','abalone')

    pipeM.set_params(**madelon_final_params)
    pipeM.set_params(**{'MLP__early_stopping':False})               
    iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','madelon')        
    pipeA.set_params(**abalone_final_params)
    pipeA.set_params(**{'MLP__early_stopping':False})                  
    iterationLC(pipeA,abalone_trgX,abalone_trgY,abalone_tstX,abalone_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','abalone')                

    pipeM.set_params(**madelon_OF_params)
    pipeM.set_params(**{'MLP__early_stopping':False})                  
    iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','madelon')        
    pipeA.set_params(**abalone_OF_params)
    pipeA.set_params(**{'MLP__early_stopping':False})               
    iterationLC(pipeA,abalone_trgX,abalone_trgY,abalone_tstX,abalone_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','abalone')                

if __name__ == "__main__":
    main()