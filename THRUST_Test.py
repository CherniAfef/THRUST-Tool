# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:00:10 2022

@author: Afef Cherni, Roxane Bertrand, Magalie Ochs
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
import THRUST_Tool as THRUST
import os as os

chemin = os.getcwd() + '/'

# paramètres du modèle POM
pomtitle = np.loadtxt("PomData/list.txt", dtype='str')
Xpom = np.loadtxt("PomData/X_Aus_mvtTete.txt")
ypom = np.loadtxt("PomData/y_Aus_mvtTete.txt")
data = pd.DataFrame(Xpom, columns=pomtitle[1:-1])

# paramètres de séquence à modéliser
for numseq in [1, 2, 3, 4, 5]:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Sequence: ', str(numseq), '        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    for loc in ["Alice", "Kevin", "Emma", "Marjorie", "Ellie"]:      
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       ', str(loc), '        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        title_seq = 'Seq' + str(numseq)  + '_' + str(loc) + '_' + "brute"

        titre = chemin  + 'Data_brute/' + title_seq + '.csv'
        tab_orig = pd.read_csv(titre)
        # paramètres de méthode à utiliser et classe
        for class_s in [0, 1]:
            
            #### Perusasion Model ---------------------------------------------------------------- 
            Tab_persuasion = THRUST.Main_persuasion(data, ypom, numseq, loc, class_s, tab_orig)
            title_seq = 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s)
            # save
            if not os.path.exists(chemin + '/Data_persuasion'):
                os.mkdir(chemin + '/' + 'Data_persuasion/')
            Tab_persuasion.to_csv(chemin  + '/Data_persuasion/' + title_seq + '.csv', index=False)
            print('--------- Modelization of Sequence: ', ' class: ', str(class_s), ' is done')   
    
            #### Perusasion Greta ---------------------------------------------------------------- 
            Tab_greta = THRUST.Main_Greta(Tab_persuasion, numseq, loc, class_s)
            # save
            if not os.path.exists(chemin + 'DataGreta/'):
                os.mkdir(chemin + 'DataGreta/')
            if not os.path.exists(chemin + '/DataGreta/' + 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s)):
                os.mkdir(chemin + '/DataGreta/' + 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s) +'/')
            Tab_greta.to_csv(chemin + '/DataGreta/' + 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s) + '/' + str(class_s) + '.csv', index=False)
            print('--------- Greta Data:               ', ' class: ', str(class_s), ' is ready to be simulated')   
            
