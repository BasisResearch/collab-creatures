# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:52:17 2023

@author: admin
"""

import numpy as np
import gridworld_utils as util

class TreeEnvironment(object):
    def __init__(self): 
        return
    
class BCCH_species(object):
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor): 
        self.discount_factor = discount_factor
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob)
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # in which locations do I see a neighboring agent?
        return 
    
    # def reward_function(self, features):
    #     # compute weighted sum of features
    #     xloc_allagents, yloc_allagents = util.loc1Dto2D(arr_loc_id_allagents, edge_size)
    #     xloc_self, yloc_self = util.loc1Dto2D(prev_state, edge_size)
    #     f_otheragents_2d = agent.compute_reward_otheragents(xloc_allagents, yloc_allagents, xloc_self, yloc_self, edge_size)
    #     f_otheragents_1d = np.reshape(f_otheragents_2d, (N_states, 1))
    #     sum_weighted_features = c_food * phi_food   + c_otheragents * f_otheragents_1d
    #     return sum_weighted_features


class TUTI_species(object):
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor): 
        self.discount_factor = discount_factor
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob)
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # in which locations do I see a neighboring agent?
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
        xloc_allagents, yloc_allagents = util.loc1Dto2D(arr_loc_id_allagents, edge_size)
        xloc_self, yloc_self = util.loc1Dto2D(prev_state, edge_size)
        social_reward_map = agent.compute_reward_otheragents(xloc_allagents, yloc_allagents, xloc_self, yloc_self, edge_size)
        social_reward_arr = np.reshape(social_reward_map, (N_states, 1))
        sum_weighted_features = c_food * phi_food + c_neighbors * social_reward_arr
        
        '''
        reward_map = np.zeros([edge_size, edge_size])
        
        N_neighbors = len(xloc_neighbors)  # number of neighboring agents
        
        for k in range(N_neighbors): 
            col = xloc_neighbors[k] 
            row = yloc_neighbors[k]
            # create a donut-shaped reward field centered on the neighbor

            reward_map[row-1 : row+2 , col-1 : col+2] = 0.3
            reward_map[row + 0, col + 0] = -2   # give the neighbor some personal space
            
        
        # make your current state neutral
        reward_map[yloc_self-1 : yloc_self+2 , xloc_self-1 : xloc_self+2] = 0
        reward_map[yloc_self, xloc_self] = 0 
        
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