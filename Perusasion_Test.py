# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:00:10 2022

@author: Afef Cherni
"""

import os as os
import numpy as np
import pandas as pd
from Code import THRUST_Tool as THRUST

path = os.getcwd() + '/'

# POM parameters
pomtitle = np.loadtxt("Pom_Data/list.txt", dtype='str')
Xpom = np.loadtxt("Pom_Data/X_Aus_mvtTete.txt")
ypom = np.loadtxt("Pom_Data/y_Aus_mvtTete.txt")
data = pd.DataFrame(Xpom, columns=pomtitle[1:-1])

# modelization
for numseq in [1, 2, 3, 4, 5]:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Sequence: ', str(numseq), '        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    for loc in ["Alice", "Kevin", "Emma", "Marjorie", "Ellie"]: #speaker  
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       ', str(loc), '        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        title_seq = 'Seq' + str(numseq)  + '_' + str(loc) + '_' + "brute" #title of original sequence
        title = path  + 'Original_Data/' + title_seq + '.csv' 
        tab_orig = pd.read_csv(title) #upload the original sequence
        
        for class_s in [0, 1]: #class persuasive (1) or not persuavise (0)
            
            #### Perusasion Model ---------------------------------------------------------------- 
            Tab_persuasion = THRUST.Main_persuasion(data, ypom, numseq, loc, class_s, tab_orig)
            title_seq = 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s)
            # save
            if not os.path.exists(path + '/Persuasion_Data'):
                os.mkdir(path + '/' + 'Persuasion_Data/')
            Tab_persuasion.to_csv(path  + '/Persuasion_Data/' + title_seq + '.csv', index=False)
            print('--------- Modelization of Sequence: ', ' class: ', str(class_s), ' is done')   
    
            #### Perusasion Greta ---------------------------------------------------------------- 
            Tab_greta = THRUST.Main_Greta(Tab_persuasion, numseq, loc, class_s)
            # save
            if not os.path.exists(path + 'Greta_Data/'):
                os.mkdir(path + 'Greta_Data/')
            if not os.path.exists(path + '/Greta_Data/' + 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s)):
                os.mkdir(path + '/Greta_Data/' + 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s) +'/')
            Tab_greta.to_csv(path + '/Greta_Data/' + 'Seq' + str(numseq) + '_' + str(loc) + '_class' + str(class_s) + '/' + str(class_s) + '.csv', index=False)
            print('--------- Greta Data:               ', ' class: ', str(class_s), ' is ready to be simulated with Greta Framework')   
            
