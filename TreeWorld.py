# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:52:17 2023

@author: admin
"""

import numpy as np

class TreeEnvironment(object):
    def __init__(self): 
        return

class SimpleAgent(object): 
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor): 
        self.discount_factor = discount_factor
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob)
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # in which locations do I see a neighboring agent?
        return 

    #  
    def compute_reward_otheragents(self, xloc_neighbors, yloc_neighbors, xloc_self, yloc_self, edge_size):
        '''
        Parameters
        ----------
        xloc_neighbors : array of x coordinates of each neighboring agent 
        yloc_neighbors : array of y coordinates of each neighboring agent 
        xloc_self : scalar x coordinate of this agent 
        yloc_self : scalar y coordinate of this agent 
        edge_size : edge size of this gridworld environment 

        Returns
        -------
        reward_map :  reward heat map (edge_size, edge_size)
        
        Example calling from main script: 
        ------
        xloc_allagents, yloc_allagents = loc1Dto2D(arr_loc_id_allagents)
        xloc_self, yloc_selc = loc1Dto2D(arr_loc_id_allagents)
        social_reward_map = agent.compute_reward_otheragents(xloc_allagents, yloc_allagents, xloc_self, yloc_self, edge_size)
        '''
        reward_map = np.zeros([edge_size, edge_size])
        
        N_neighbors = len(xloc_neighbors)  # number of neighboring agents
        
        for k in range(N_neighbors): 
            x = xloc_neighbors[k] 
            y = yloc_neighbors[k]
            # create a donut-shaped reward field centered on the neighbor
            reward_map[x-1 : x+1 , y-1 : y+1] = 1  
            reward_map[x + 0, y + 0] = -1    # give the neighbor some personal space
            
        return reward_map
        
        # *********** Alternative strategies ****************************
        
        # 1) 
        # for each neighbor: (for now, assume all neighbors are same species)
        # convert the neighbor's location ID to x,y coordinates 
        # compute euclidean distance between me and my neighbor using x,y coordinates
        # compute reward as a function of euclidean distance 
        # convert back to location IDs and then reshape to create heat map 
        #
        # 2) 
        # # compute social reward for each location 
        # for x in range(edge_size):
        #   for y in range(edge_size):
        #     for i in range(N_agents): 
        #         x_agent, y_agent = arr_loc_id_allagents[i]
        #         y_agent = 
        #         dist = np.sqrt( (x - x_agent)**2 + (y - y_agent)**2 )
        #         social_reward_map[i, j] = -np.cos(dist)