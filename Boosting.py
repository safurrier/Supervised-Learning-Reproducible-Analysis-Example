# -*- coding: utf-8 -*-



import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)



def main():

    cars = pd.read_hdf('data/processed/datasets.hdf','cars')        
    carsX = cars.drop('Class',1).copy().values
    carsY = cars['Class'].copy().values

    madelon = pd.read_hdf('data/processed/datasets.hdf','madelon')        
    madelonX = madelon.drop('Class',1).copy().values
    madelonY = madelon['Class'].copy().values

    alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


    cars_trgX, cars_tstX, cars_trgY, cars_tstY = ms.train_test_split(carsX, carsY, test_size=0.3, random_state=0,stratify=carsY)     
    madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     



    madelon_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
    cars_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
    OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
    #paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
    paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
              'Boost__base_estimator__alpha':alphas}
    #paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
    #           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

    paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
               'Boost__base_estimator__alpha':alphas}


    madelon_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=madelon_base,random_state=55)
    cars_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=cars_base,random_state=55)
    OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                     ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                     ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                     ('Boost',madelon_booster)])

    pipeA = Pipeline([('Scale',StandardScaler()),                
                     ('Boost',cars_booster)])

    #
    madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,paramsM,'Boost','madelon')        
    cars_clf = basicResults(pipeA,cars_trgX,cars_trgY,cars_tstX,cars_tstY,paramsA,'Boost','cars')        

    #
    #
    #madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
    #cars_final_params = {'n_estimators': 10, 'learning_rate': 1}
    #OF_params = {'learning_rate':1}

    madelon_final_params = madelon_clf.best_params_
    cars_final_params = cars_clf.best_params_
    OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

    ##
    pipeM.set_params(**madelon_final_params)
    pipeA.set_params(**cars_final_params)
    makeTimingCurve(madelonX,madelonY,pipeM,'Boost','madelon')
    makeTimingCurve(carsX,carsY,pipeA,'Boost','cars')
    #
    pipeM.set_params(**madelon_final_params)
    iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','madelon')        
    pipeA.set_params(**cars_final_params)
    iterationLC(pipeA,cars_trgX,cars_trgY,cars_tstX,cars_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','cars')                
    pipeM.set_params(**OF_params)
    iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','madelon')                
    pipeA.set_params(**OF_params)
    iterationLC(pipeA,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','cars')                

if __name__ == "__main__":
    main()
