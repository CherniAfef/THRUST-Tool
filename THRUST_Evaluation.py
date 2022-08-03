# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:40:49 2022

@author: Afef Cherni, Roxane Bertrand, Magalie Ochs
"""

import numpy as np
import pandas as pd
import os as os
from joblib import load

chemin = os.getcwd() + '/'

listGreta_test = ['pose_Rx' , 'pose_Ry' , 'pose_Rz' ,
            'AU01_r', 'AU02_r' , 'AU04_r', 'AU05_r', 'AU06_r' , 'AU07_r',  'AU12_r',]



model_rf = load('best_rf.joblib')
Seq_list = [1, 2, 3, 4, 5]
Loc_list = ["Alice", "Kevin", "Emma", "Marjorie", "Ellie"]
for class_s in [ 0, 1 ]:
    prediction_score = 0
    for numseq in Seq_list:
        print("--------------------- Sequence ", numseq)
        for loc in Loc_list:
    
            title_seq =  'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s) + '/' + str(class_s)
            T = pd.read_csv(chemin  + 'Data_Greta/' + title_seq + '.csv')
            data = T[listGreta_test] 
            df0 = pd.DataFrame()
            for i in listGreta_test:
                moyvalue = np.mean(data[i])
                medianvalue = np.median(data[i])
                minvalue = np.min(data[i])
                maxvalue = np.max(data[i])
                stdvalue = np.std(data[i])
                varvalue = np.var(data[i])
    
                data1 = {'moy_'+str(i):moyvalue, 'median_'+str(i):medianvalue, 'min_'+str(i):minvalue, 'max_'+str(i):maxvalue, 'std_'+str(i):stdvalue, 'var_'+str(i):varvalue}
                df1 = pd.DataFrame(index=[1,], data=data1)
                newdf1 = pd.concat([df0, df1], axis = 1)
                df0 = newdf1
    
            X_test = np.array(df0)
            y_pred = model_rf.predict(X_test)    
            if y_pred[0] == class_s:
                prediction_score += 1
            print(loc, ": prediction = ", y_pred, "original class = ", class_s)
        del data
        
    print('                                                                           ')
    print('                                                                           ')
    print('                                                                           ')
    print('Using the RF classifier, we obtain', prediction_score , 'correct predictions %', len(Seq_list)*len(Loc_list), 'tests')
    print('                                                                           ')
    if prediction_score > len(Seq_list)*len(Loc_list) - 3:
        print(' ---------------         NICE WORK!  ^_^               -------------------') 
    print('                                                                           ')
    print('                                                                           ')
    print('                                                                           ')
