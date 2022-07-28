# THRUST-Tool

This repository is associated to the manuscript 

* ### From neutral human face to persuasive virtual face: a new automatic tool to generate a persuasive attitude
* #### Afef Cherni, Roxane Bertrand and Magalie Ochs


## Abstract
In order to motivate the user to change her/his behavior or attitudes, for instance to practice physical activities to improve her/his well-being, virtual agents should have persuasive capabilities.
The persuasiveness of the virtual agent not only depends on its speech but also on its non-verbal behavioral cues. In this paper, we propose the new tool called THRUST (from neuTral Human face to peRsUaSive virTual face), to automatically generate the head movements and facial expressions of a persuasive virtual character from a video of a human. Combining a machine learning approach on a corpus of persuasive human speech and a convolution-based
method, we propose a model, based on real data of persuasive human message, that transforms the non-verbal behavior of the human expressed in a video to a persuasive non-verbal behavior replicated on a virtual face. 


![Model](https://user-images.githubusercontent.com/24696985/181495525-b33a34fd-f8cc-492f-9c3f-a804bda51ed1.PNG)



This program HAS NOT been tested intensively, it is believed to do what it is supposed to do, However, you are welcome to check it on your own data.


    Authors : Afef Cherni,  Roxane Bertrand and Magalie Ochs 
    Contact: cherni.afef@univ-amu.fr
    Version : 1.0   Date : July 2016

## What we need to use THRUST ?
Since our model take into account the 

## How to use THRUST ?
* 1- Clone the repository to retrieve all files from the THRUST Project
* 2- Check if you have OriginalData folder, POMData folder, THRUST_Tool.py, THRUST_Test and THRUST_Evaluation
* 3- To test our proposed tool, you should:

a) Run THRUST_Test: this code uses the input $U_i$ from OriginalData folder and the references $M_i$ from POMData folder in order to create a new folder called Data_persuasion that contains the output ($W_i$) of our model. This step ensures the transformation of the attitude from $Neutral$ (given by $U_i$) to "Persuasive" (given by $W_i$).

b) Run THRUST_Evaluation: this script checks if the output of our THRUST model are classified as "Persuasive" or "Neutral". For that, we need to download the Random Forest classifier (optimized and stored as "best_rf.joblib") 
