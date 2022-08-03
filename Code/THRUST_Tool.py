# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:00:10 2022

@author: Afef Cherni, Roxane Bertrand, Magalie Ochs
"""


import os as os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
import scipy

pd.options.mode.chained_assignment = None  

path = os.getcwd() + '/'

# Action Units features (AUs) & head movement (x,y,z - axis)
def modelization(data_corpus, score_corpus, class_s, data_avatar, w, flag_show=False):
    
    # 1) AUs
    data_corpus = data_corpus.iloc[score_corpus==class_s] 
    model =  data_corpus.loc[:, :] 
    tab = data_avatar.copy()
    
    for var in [1, 2, 4, 5, 6, 7, 12,]:
        #pom
        var_pom = 'moy_AU' + str(var)
        #avatar
        if var < 10:
            var_avatar = ' AU0' + str(var) + '_r'

        else:
            var_avatar = ' AU' + str(var) + '_r'
        
            
        # 1) re-samlling the pom model
        model[var_pom] = (model[var_pom] + 10 ) / 4
        model_res = signal.resample(np.array(model[var_pom]), len(data_avatar))

        
        # 2) convolution on w-window

        k = 0
        while k < np.shape(model_res)[0]:      
            if max(data_avatar[var_avatar][k:k+w]) - min(data_avatar[var_avatar][k:k+w]) < 0.05:
                tab[var_avatar][k:k+w] = model_res[k:k+w] 
            else:
                tab[var_avatar][k:k+w] = np.convolve(data_avatar[var_avatar][k:k+w], model_res[k:k+w] , mode="same") 
            k += w
        tab[var_avatar].loc[tab[var_avatar] > 5] = 5
        tab[var_avatar].loc[tab[var_avatar] < 0] = 0

    # 2) head movements
    for var in ['x', 'y', 'z']:
        var_model = 'moy_pose_R' + str(var)
        var_avatar = ' pose_R' + str(var)

        # 1) resample the model
        model_res = signal.resample(np.array(model[var_pom]), len(data_avatar)) 

        # 2) convolution on w-window
        k = 0
        while k < np.shape(model_res)[0]:        

            if max(data_avatar[var_avatar][k:k+w]) - min(data_avatar[var_avatar][k:k+w]) < 0.05:
                tab[var_avatar][k:k+w] = model_res[k:k+w]
            else:
                tab[var_avatar][k:k+w] = np.convolve(data_avatar[var_avatar][k:k+w], model_res[k:k+w] , mode="same")  
            tab[var_avatar].loc[tab[var_avatar] > 5] = 5
            tab[var_avatar].loc[tab[var_avatar] < 0] = 0
            k += w
        tab[var_avatar] = tab[var_avatar] / np.linalg.norm(tab[var_avatar])

    if flag_show == True:
        i = 1
        print('-----------------  AUs modelization display enabled- ----------------- ')        
        plt.figure(figsize=(12,8))
        for var in [1, 2, 4, 5, 6, 7, 12,]:
            if var < 10:
                var_avatar = ' AU0' + str(var) + '_r'
            else:
                var_avatar = ' AU' + str(var) + '_r'
            var_model = 'moy_AU' + str(var)
            plt.subplot(7,1,i)
            plt.plot(data_avatar[var_avatar], 'g', label="Avatar")
            plt.plot(model[var_model], 'b--', label="POM")
            plt.plot(tab[var_avatar], 'r-.', label="Conv")
            plt.legend()
            plt.title('AU_' + str(var))
            i += 1

        j = 1
        print('-----------------  Head movement modelization display enabled- ----------------- ')        
        plt.figure(figsize=(12,8))
        for var in ['x', 'y', 'z']:
            var_avatar = ' pose_R' + str(var)
            var_model = 'moy_pose_R' + str(var)
            plt.subplot(3,1,j)
            plt.plot(data_avatar[var_avatar], 'g', label="Avatar")
            plt.plot(model[var_model], 'b--', label="POM")
            plt.plot(tab[var_avatar], 'r-.', label="Conv")
            plt.legend()
            plt.title('pose_' + str(var))
            j += 1
    plt.show()
    return tab


def post_treatment(tab, w, flag_show=False):
    
    tab_au1 = tab[' AU01_r'].copy()
    tab_au2 = tab[' AU02_r'].copy()
    tab_au4 = tab[' AU04_r'].copy()
    tab_au6 = tab[' AU06_r'].copy()
    tab_au5 = tab[' AU05_r'].copy()
    tab_au7 = tab[' AU07_r'].copy()
    tab_au12 = tab[' AU12_r'].copy()

    k = 0
    while k < np.shape(tab_au12)[0]:
        mean_au7 = np.mean(tab_au7[k:k+w])
        mean_au6 = np.mean(tab_au6[k:k+w])
        mean_au5 = np.mean(tab_au5[k:k+w])
        mean_au4 = np.mean(tab_au4[k:k+w])
        mean_au2 = np.mean(tab_au2[k:k+w])
        mean_au1 = np.mean(tab_au1[k:k+w])

        if mean_au1 > mean_au2:
            tab_au2[k:k+w] = 0
        else:
            tab_au1[k:k+w] = 0

        if mean_au1 > mean_au4 or mean_au2 > mean_au4:
            tab_au4[k:k+w] = 0

        if mean_au4 > mean_au6:
            tab_au4[k:k+w] = 0

        tab_au7[k:k+w] = 3 * tab_au7[k:k+w] / np.linalg.norm(tab_au7[k:k+w])
        tab_au5[k:k+w] = 3 * tab_au5[k:k+w] / np.linalg.norm(tab_au5[k:k+w])    
        k += w

    # updates
    tab[' AU01_r'] = tab_au1 
    tab[' AU02_r'] = tab_au2
    tab[' AU04_r'] = tab_au4
    tab[' AU05_r'] = tab_au5
    tab[' AU06_r'] = tab_au6
    tab[' AU07_r'] = tab_au7
    tab[' AU12_r'] = tab_au12    
        
    if flag_show == True:
        print('----------------- Post-treatement display enabled----------------- ')
        plt.figure(figsize=(12,8))
        plt.subplot(3,1,1)
        plt.plot(tab_au1, 'r', label='au01+pt')
        plt.plot(tab_au2, 'b', label='au02+pt')
        plt.plot(tab_au4, 'g', label='au04+pt')
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(tab[' AU07_r'], 'b--', label='au07 orig')
        plt.plot(tab_au7, 'r', label='au07+pt')
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(tab[' AU05_r'], 'b--', label='au05 orig')
        plt.plot(tab_au5, 'r', label='au05+pt')
        plt.legend()
    plt.show()
    
    return tab


def smoothing(tab, flag_show=False):
    i = 1
    ttab = tab.copy()
    for var in [1, 2, 4, 5, 6, 7, 12,]:
        if var < 10:
            var_avatar = ' AU0' + str(var) + '_r'
        else:
            var_avatar = ' AU' + str(var) + '_r'

        ttab[var_avatar] = scipy.signal.savgol_filter(tab[var_avatar],55, 11)    

        if flag_show == True:
            if i == 1:
                print('----------------- Smoothing display enabled ----------------- ')
                plt.figure(figsize=(12,12))
            plt.subplot(8,1,i)
            plt.plot(tab[var_avatar], 'r-', label="before smooting")
            plt.plot(ttab[var_avatar], 'g--', label="after smooting")
            plt.legend()
            plt.title('AU_'+ str(var))
            i += 1 
        plt.show()

    return ttab


def Main_persuasion(data_pom, score_pom, num, loc, class_s, tab_openface):    
    if class_s == 1:
        T1 = modelization(data_corpus=data_pom, score_corpus=score_pom, class_s=class_s, data_avatar=tab_openface, w=7, flag_show=False)
        
        T2 = post_treatment(T1, w=7, flag_show=False)
        
        T3 = smoothing(T2, flag_show=False)
        
    elif class_s == 0:
        T1 = modelization(data_corpus=data_pom, score_corpus=score_pom, class_s=class_s, data_avatar=tab_openface, w=3, flag_show=False)
        T3 = T1
    else:
        print('Error, your class schould be 0 or 1!')
    return T3

def Main_Greta(data, numseq, loc, class_s):
    listGreta = [' timestamp', ' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' gaze_angle_x', ' gaze_angle_y',
            ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx' , ' pose_Ry' , ' pose_Rz' ,
             ' AU01_r', ' AU02_r' , ' AU04_r', ' AU05_r', ' AU06_r' , ' AU07_r', ' AU09_r' , ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', 
             ' AU17_r', ' AU20_r', ' AU23_r' , ' AU25_r', ' AU26_r' , ' AU45_r']
    
    listG = ['timestamp', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
            'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx' , 'pose_Ry' , 'pose_Rz' ,
             'AU01_r', 'AU02_r' , 'AU04_r', 'AU05_r', 'AU06_r' , 'AU07_r', 'AU09_r' , 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 
             'AU17_r', 'AU20_r', 'AU23_r' , 'AU25_r', 'AU26_r' , 'AU45_r']
 
    T = data[listGreta]
    T.columns = listG
    T['timestamp'].loc[1] = T['timestamp'].loc[0] + 0.024
    return T

