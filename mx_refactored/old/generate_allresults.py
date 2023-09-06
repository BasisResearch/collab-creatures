import numpy as np
import pandas as pd


# PART 1: Generate raw data by running the forward model 

for sim in range(N_sims): # for each simulation (row in metadata table)

    #extract parameters from that row of metadataDF
    param_list

    #run the forward model N_runs times with parameters from that row 
    for run in range(N_runs):

        # 1. run the forward model for N_frames time steps # TO DO
        bird_locsDF, reward_locsDF = run_model(param_list, N_frames)

        # Save the raw data in .csv format in a Google Drive with folder names reflecting the parameters and their values 
        #   Raw data directory on Marjorie's local computer: C:\Users\admin\Dropbox\Code\Basis-code\data\communication_model_raw_data
        #   Make sure to include the parameter values in the name of the file so we can search for the appropriate file later on
        parent_folder = date # organize simulations by date???
        birdfilename = '_sim' + str(sim) + '_run' + str(run) + 'birdlocs'
        rewardfilename = '_sim' + str(sim) + '_run' + str(run) + 'rewardlocs'

        bird_locsDF.to_csv(birdfilename + ".csv")
        reward_locsDF.to_csc(rewardfilename + ".csv")



# PART 2: Compute success measures from raw data
# It's possible that we may not need a for loop. 

allresultsDF = pd.DataFrame()

for sim in range(N_sims): # for each simulation (row in metadata table)

    #extract parameters from that row of metadataDF
    param_list

    #run the forward model N_runs times with parameters from that row 
    for run in range(N_runs):

        # 1. Fetch raw data from folder 
        datapath_str = 
        birdlocs = pd.read_csv(datapath_str)

        # 2. compute success metrics using the raw data from that run
        # 2.1. Time to first food, averaged across birds in the group
        mean_time_to_food_scalar = compute_avg_time_to_first_food(bird_locsDF, reward_locsDF)
        # 2.2. Number of birds that failed to find food during the sim 
        mean_num_failed_scalar = compute_num_failed(bird_locsDF, reward_locsDF)

        

        # # Add the success metrics to results data frame 
        # # TO DO: figure out how to combine success metrics and metadata into a new data frame
        # # maybe want to save in arrays first and then integrate the arrays into a new data frame 
        # or just save as dataframes
        # mean_time_to_food_arr[run] = mean_time_to_food_scalar
        # mean_num_failed_arr[run] = mean_num_failed_scalar


# Part 3: Populate allresultsDF with success metrics stich together the data column-wise
allresultsDF = pd.concat([metadataDF, successmeasuresDF], axis=1) # column-wise concatenation
    
# Save the allresults dataframe somewhere
