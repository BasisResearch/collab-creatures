import numpy as np
import pandas as pd


# for each simulation type, run the forward model N times
#   for each run,
#       run the forward model
#       collect bird locations
#       collect reward locations
#       compute success measures (e.g. time to first food)
#       save bird locations and reward locations in Google Drive
#       generate dataframe containing success measures for each run along with the metadata 

for sim in range(N_sims): # for each simulation

    #extract parameters from that row of metadataDF
    param_list

    #run the forward model N_runs times with parameters from that row 
    for run in range(N_runs):

        # 1. run the forward model for N_frames time steps
        bird_locsDF, reward_locsDF = run_model(param_list, N_frames)

        # 2. compute success metrics using the raw data from that run
        # 2.1. Time to first food, averaged across birds in the group
        mean_time_to_food_scalar = compute_avg_time_to_first_food(bird_locsDF, reward_locsDF)
        # 2.2. Number of birds that failed to find food during the sim 
        mean_num_failed_scalar = compute_num_failed(bird_locsDF, reward_locsDF)

        # Add the success metrics to results data frame 
        # TO DO: figure out how to combine success metrics and metadata into a new data frame
        # maybe want to save in arrays first and then integrate the arrays into a new data frame
        mean_time_to_food_arr[run] = mean_time_to_food_scalar
        mean_num_failed_arr[run] = mean_num_failed_scalar

        # Save the raw data in .csv format in a Google Drive with folder names reflecting the parameters and their values 
        #   Raw data directory on Marjorie's local computer: C:\Users\admin\Dropbox\Code\Basis-code\data\communication_model_raw_data
        #   Make sure to include the parameter values in the name of the file so we can search for the appropriate file later on
        bird_locsDF.to_csv()
        reward_locsDF.to_csc()

    

    # Save the allresults dataframe somewhere
