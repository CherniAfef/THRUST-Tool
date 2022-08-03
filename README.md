# THRUST-Tool

This repository is associated to the manuscript 

* ### From neutral human face to persuasive virtual face: a new automatic tool to generate a persuasive attitude
* #### Afef Cherni, Roxane Bertrand, Magalie Ochs


## Abstract
In order to motivate the user to change her/his behavior or attitudes, for instance to practice physical activities to improve her/his well-being, virtual agents should have persuasive capabilities.
The persuasiveness of the virtual agent not only depends on its speech but also on its non-verbal behavioral cues. In this paper, we propose the new tool called THRUST (from neuTral Human face to peRsUaSive virTual face), to automatically generate the head movements and facial expressions of a persuasive virtual character from a video of a human. Combining a machine learning approach on a corpus of persuasive human speech and a convolution-based
method, we propose a model, based on real data of persuasive human message, that transforms the non-verbal behavior of the human expressed in a video to a persuasive non-verbal behavior replicated on a virtual face. 

## Global architecture 
To illustrate THRUST model, we propose the overview bellow:

![Model](https://user-images.githubusercontent.com/24696985/181495525-b33a34fd-f8cc-492f-9c3f-a804bda51ed1.PNG)

In nutshell, THRUST model takes as input a video of a  speaker talking  about a specific topic in a neutral way. The input is not limited to real-time data video, it can be webcam video, recorded video files or sequences of images. The important aspect is  to be able to extract the facial landmarks, the  head poses, the eye gaze and the facial Action Unit (AUs) from the video wich will be the input $U_i$ of our Model. Then, it ensures the transformation of face ad head movement from neutral to persuasive attitude noted $W_i$. For that, it uses the refere ce $M_i$ computed based to POM corpus (Details are given in the paper).
We simulate the output variables set $W_i$ on the embodied conversational agent Greta, and generates the target video.


This program HAS NOT been tested intensively, it is believed to do what it is supposed to do, However, you are welcome to check it if you have own corpus ad own data.


    Authors : Afef Cherni,  Roxane Bertrand and Magalie Ochs 
    Contact: cherni.afef@univ-amu.fr
    Version : 1.0   Date : July 2022

## How to use THRUST ?
* 1- Clone the repository to retrieve all files from the THRUST Project
* 2- Check if you have Original_Data folder, POM_Data folder, THRUST_Tool.py, THRUST_Test and THRUST_Evaluation
* 3- To test our proposed tool, you should:

a) Run THRUST_Test: this code uses the input $U_i$ from Original_Data folder and the references $M_i$ from POM_Data folder in order to create a new folder called Perusasion_Data that contains the output ($W_i$) of our model. This step ensures the transformation of the attitude from $Neutral$ (given by $U_i$) to "Persuasive" (given by $W_i$).

b) Run THRUST_Evaluation: this script checks if the output of our THRUST model are classified as persuasive or neutral. For that, we need to download the Random Forest classifier (optimized and stored as "best_rf.joblib") 

## Do you have your own data?
* If you would like to test our THRUST tool with your own video, you should just extract the head poses and facial action unit (you can use for this OpenFace toolkit) and save the extracted file (in .csv format) in OriginalData folder.
Preferably, name your file like the examples given, otherwise you have to modify the code, it's up to you to do what is necessary in this case ;-)

## What we need to use THRUST ?
* 1- Python realease (https://www.python.org/downloads/)
* 2- Greta platform to generate video with embodied conversational agent (https://github.com/isir/greta)
* 3- OpenFace toolkit (https://github.com/TadasBaltrusaitis/OpenFace)


## If you want !
* 1- To explore POM corpus (https://github.com/eusip/POM/)
* 2- To find the results presented in the article (ML_test.ipynb)

## Demonstration
