# AM205 Final Project
#### Contributors: Xincheng Tan, Yaxin Lei
Topic: Gaussian graphical model (GGM) and randomized linear algebra (RLA) for estimating the functional connectivity 
among a neuron population

In particular, we aim to apply RLA matrix sketching techniques and apply them in the GGM estimation pipeline. 
We would like to compare the difference in the estimated result between regular Graphical Lasso (glasso) and RLA-infused glasso. 


## Report
Please see **AM205_Final_Report.pdf** for our final results and conclusions.


## Dataset
The datasets we use for this project is published by Carsen Stringer at 
https://janelia.figshare.com/articles/dataset/Recordings_of_10k_neurons_in_V1_during_drifting_gratings/6214019.  

Note: In order to run the jupyter notebook code, please download the whole dataset locally and specify the correct path.


## Code
The main code for functional connectivity estimation is in **Neuronal Functional Connectivity Estimation.ipynb**.
The code for randomized linear algebra matrix multiplication is in **RLA_tests_graphs.ipynb**.  
  
The rest of the .py files provide helper functions to load the dataset,
apply the Gaussian graphical model with and without RLA, and plot the estimated results under various evaluation metrics.
