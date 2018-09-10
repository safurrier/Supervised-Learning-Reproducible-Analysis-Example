# -*- coding: utf-8 -*-


import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
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


    d = abaloneX.shape[1]
    hiddens_abalone = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    alphas = [10**-x for x in np.arange(1,9.01,1/2)]
    d = madelonX.shape[1]
    hiddens_madelon = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]


    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('KNN',knnC())])  

    pipeA = Pipeline([('Scale',StandardScaler()),                
                     ('KNN',knnC())])  



    params_madelon= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
    params_abalone= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

    madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params_madelon,'KNN','madelon')        
    abalone_clf = basicResults(pipeA,abalone_trgX,abalone_trgY,abalone_tstX,abalone_tstY,params_abalone,'KNN','abalone')        


    #madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
    #abalone_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
    madelon_final_params=madelon_clf.best_params_
    abalone_final_params=abalone_clf.best_params_



    pipeM.set_params(**madelon_final_params)
    makeTimingCurve(madelonX,madelonY,pipeM,'KNN','madelon')
    pipeA.set_params(**abalone_final_params)
    makeTimingCurve(abaloneX,abaloneY,pipeA,'KNN','abalone')
    
if __name__ == "__main__":
    main()