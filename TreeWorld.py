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

class SimpleAgent(object): 
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor, energy_init=50): 
        self.discount_factor = discount_factor # scalar between 0 and 1 
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob) # (N_states, N_states) matrix 
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # indicator  vector showing which locations do I see a neighboring agent?
        self.energy_total = energy_init # scalar quantity (calories)
        self.energy_trajectory = np.zeros([N_timesteps]) # not sure if we need this to be inside the class
        self.energy_trajectory[0] = energy_init
        self.sight_radius = 7
        return 

    def compute_visible_locations(self, xloc_self, yloc_self, edge_size):
        phi_visible_mat = np.zeros([edge_size, edge_size])
        x_arr = np.arange(edge_size)
        y_arr = np.arange(edge_size)
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='xy')
        phi_visible_mat = np.sqrt((x_grid - xloc_self)**2 + (y_grid - yloc_self)**2) <= self.sight_radius
        
        return phi_visible_mat
    
    
    def reward_function_otheragents(self, xloc_neighbors, yloc_neighbors, xloc_self, yloc_self, edge_size):
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

            reward_map[row-1 : row+2 , col-1 : col+2] = 0
            reward_map[row + 0, col + 0] = 1  # give the neighbor some personal space
            
        
        # # make your current state neutral
        # reward_map[yloc_self-1 : yloc_self+2 , xloc_self-1 : xloc_self+2] = 0
        # reward_map[yloc_self, xloc_self] = 0 
        
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
        
        
class BCCH_species(object):
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor): 
        self.discount_factor = discount_factor
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob)
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # in which locations do I see a neighboring agent?
        
        self.food_depletion_rate = 0.3
        self.feature_weights = [2, 0.5] # c weights for [food, other agents]
        
        return 
    
    def reward_function_otheragents(self, xloc_neighbors, yloc_neighbors, xloc_self, yloc_self, edge_size):

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
    
    def value_function(self, sum_weighted_features):
        value = self.SR @ sum_weighted_features
        return value 
    
    
    def policy(self, prev_state, value, T_eligible, phi_agents):
        # eligible states are those specified by the transition matrix and states not occupied by other agents
        eligible_states_id = np.nonzero(T_eligible[:, prev_state] * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
        value_eligible = value[eligible_states_id].flatten()   # value of eligible states plus some noise 
        next_state = eligible_states_id[np.argmax(value_eligible)]  # DETERMINISTIC POLICY that works
        
        return next_state
        


class TUTI_species(object):
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor): 
        self.discount_factor = discount_factor
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob)
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # in which locations do I see a neighboring agent?
        
        self.food_depletion_rate = 0.5
        return 