The code in this folder is for generating simulated data from the revised communication model (see root folder file cvpr_submission_sims.py for the original model). The degree of communication in the revised model is controlled by a single parameter in the value function of the agent called c_trust.


branch: mx-cleanup
code from CVPR paper Fig. 2 is inside the root folder
    cvpr_submission_sims.py
    cvpr_submission_figures.py

refactored code is inside the folder *mx_refactored*. 
You need two scripts:
    1) generate_data_pipeline.ipynb
    2) histograms.ipynb


***** generate_data_pipeline.ipynb *****

Inputs: user decides what experimental conditions to simulate. Each experimental condition corresponds to one simulation 
Outputs: 

1) metadataDF: dataframe specifying parameters for each simulation

2) additional_meta_parameters: parameters that apply to all of the simulations the user ran

3) resultsDF: outcomes of each run of the simulation including foraging success meaures

4) each run of a simulation generates a folder containing two dataframes: birdlocsDF (location of each bird at each time) and rewardlocsDF (locations of all food rewards at each time)


***** histograms.ipynb *****

This script loads the resultsDF dataframe and runs analyses corresponding to the ones in the CVPR workshop paper Figure 2.
