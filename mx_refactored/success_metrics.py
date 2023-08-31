import numpy as np
import pandas as pd
import utils

def compute_time_to_first_reward(singlebirdDF, rewardsDF, nframes):
    '''
    Compute the time it takes for a given agent to reach its first reward.

    Inputs: 
        agentDF - dataframe containing the x and y locations of a specific agent at each time point in the simulation
        rewardsDF - dataframe containing the x and y location of each reward unit at each time point in the simulation 
    
    Output: 
        time_first -- earliest frame in the simulation where the agent is at a reward location 
    
    '''
    
    bird_xlocs = singlebirdDF['x'].to_numpy()  # extract the x locations at each frame 
    bird_ylocs = singlebirdDF['y'].to_numpy()

    time_to_first_reward = -1

    # print(ylocs)

    # birdlocs1D_alltimes_arr = utils.loc2Dto1D(xlocs, ylocs, edge_size)

    # print(birdlocs1D_alltimes_arr)

    # for each time point in the bird's trajectory, check if the bird is at a reward location
    # # for t in range(nframes): 
    for t in range(nframes): 
        # extract the rows of the rewards table that correspond to time t 
        rewards_t = rewardsDF[rewardsDF['time'] == t]
        rewards_t_xlocs = rewards_t['x'].to_numpy()  # extract the x locations at each frame 
        rewards_t_ylocs = rewards_t['y'].to_numpy()
        # rewards_t_loc1D_arr = utils.loc2Dto1D(rewards_t_xlocs, rewards_t_ylocs, 30) # array of locations occupied by a reward at time t

        # birdlocs1D_alltimes_arr = np.array([1,2,3,6,5])
        # rewards_t_loc1D_arr = np.array([9, 8, 7, 6, 5])
        # for each reward item at time t, check if the bird's location matches the location of that reward 
        for id in range(len(rewards_t)):
            # if birdlocs1D_alltimes_arr[t] == rewards_t_loc1D_arr[id]:
            if bird_xlocs[t] == rewards_t_xlocs[id] and bird_ylocs[t] == rewards_t_ylocs[id]:
                time_to_first_reward = t 
                break

        if time_to_first_reward > 0:
            break

    return time_to_first_reward