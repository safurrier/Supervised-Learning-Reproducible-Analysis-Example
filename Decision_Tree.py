# -*- coding: utf-8 -*-
"""

Script for full tests, decision tree (pruned)

"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)


def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('reports/output/DT_{}_nodecounts.csv'.format(dataset))
    
    return



    
def main():
    # Load Data       
    cars = pd.read_hdf('data/processed/datasets.hdf','cars')        
    carsX = cars.drop('Class',1).copy().values
    carsY = cars['Class'].copy().values



    madelon = pd.read_hdf('data/processed/datasets.hdf','madelon')        
    madelonX = madelon.drop('Class',1).copy().values
    madelonY = madelon['Class'].copy().values




    cars_trgX, cars_tstX, cars_trgY, cars_tstY = ms.train_test_split(carsX, carsY, test_size=0.3, random_state=0,stratify=carsY)     
    madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     

    # Search for good alphas
    alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
    #alphas=[0]
    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                     ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                     ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                     ('DT',dtclf_pruned(random_state=55))])


    pipeA = Pipeline([('Scale',StandardScaler()),                 
                     ('DT',dtclf_pruned(random_state=55))])


    params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}

    madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params,'DT','madelon')        
    cars_clf = basicResults(pipeA,cars_trgX,cars_trgY,cars_tstX,cars_tstY,params,'DT','cars')        


    #madelon_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
    #cars_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
    madelon_final_params = madelon_clf.best_params_
    cars_final_params = cars_clf.best_params_

    pipeM.set_params(**madelon_final_params)
    makeTimingCurve(madelonX,madelonY,pipeM,'DT','madelon')
    pipeA.set_params(**cars_final_params)
    makeTimingCurve(carsX,carsY,pipeA,'DT','cars')


    DTpruningVSnodes(pipeM,alphas,madelon_trgX,madelon_trgY,'madelon')
    DTpruningVSnodes(pipeA,alphas,cars_trgX,cars_trgY,'cars')
    
if __name__ == "__main__":
    main()