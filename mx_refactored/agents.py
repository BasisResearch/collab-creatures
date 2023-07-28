# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:45:41 2023

@author: admin
"""

import numpy as np
import gridworld_utils as util

class BirdAgent(object):
    def __init__(self, env, N_timesteps, agent_type='ignorer', discount_factor=0.8, energy_init=50, sight_radius=40): 
        self.env = env
        self.discount_factor = discount_factor # scalar between 0 and 1 
        self.SR = np.linalg.pinv(np.eye(env.N_states) - discount_factor * env.T_prob) # (N_states, N_states) matrix 
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([env.N_states, N_timesteps])
        self.phi_neighbors = np.zeros([env.N_states, 1]) # indicator  vector showing which locations do I see a neighboring agent?
        self.energy_total = energy_init # scalar quantity (calories)
        self.energy_trajectory = np.zeros([N_timesteps]) # not sure if we need this to be inside the class
        self.energy_trajectory[0] = energy_init
        self.times_at_food = [] # list of frames the agent is at a food location 
        self.sight_radius = sight_radius
        self.caloric_cost_per_unit_dist = 1
        
        self.caloric_cost_per_unit_dist = 1
        self.doProbabilisticPolicy = True
        self.doSoftmaxPolicy = True
        self.exploration_bias = 0.005
        
        if agent_type == 'ignorer':
            c_food_self = 0.9
            c_food_others = 0.1  # to what extent do the birds care about information from other birds?
            c_otheragents = 0
            c_group = 0
        elif agent_type == 'communicator':
            c_food_self = 0.5
            c_food_others = 0.5  # to what extent do the birds care about information from other birds?
            c_otheragents = 0
            c_group = 0
        elif agent_type == 'follower':
            c_food_self = 0.1
            c_food_others = 0.9  # to what extent do the birds care about information from other birds?
            c_otheragents = 0
            c_group = 0
                
        self.weights_list = [c_food_self, c_food_others, c_otheragents, c_group]
        
        return 
    
    # def step_agent()
    # def update_calories()
    # 
        
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

    def value_update(self, reward_vec_list):
        #weights_list is a list of scalar weights for each of the vectors in 
        #reward_vec_list. 
        #reward_vec_list is a list of N_states-length vectors predicting the 
        # amount of a partiular source of reward at each state\
        # weights_list must be the same length as reward_vec_list
        sum_weighted_features = np.zeros([len(reward_vec_list[0]), 1])
        
        for i in range(len(self.weights_list)):
            sum_weighted_features += self.weights_list[i] * reward_vec_list[i]
            
        value = self.SR @ sum_weighted_features  # (N_states, 1) vector
        
        return value
        
        
    def policy_update(self, prev_loc_1d, value):
        
        # eligible states are those specified by the transition matrix. Can constrain further to exclude states not occupied by other agents
        eligible_states_id = np.nonzero(self.env.T_eligible[:, prev_loc_1d])[0]  # * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
        value_eligible = value[eligible_states_id].flatten()  # value of eligible states plus some noise

        # ACTION SELECTION

        if self.doProbabilisticPolicy:
            if self.doSoftmaxPolicy:
                prob_arr = util.softmax(
                    value_eligible, temp=self.exploration_bias
                )
            else:
                # #sample eligible states from a categorical distribution whose shape is based on the values
                # # convert values into probabilities
                # value_eligible += 0.001 * np.random.randn(value_eligible.shape[0]) # add some noise
                # prob_arr = value_eligible - np.min(value_eligible) # shift values so they are all positive and add some noise
                prob_arr = value_eligible - np.min(value_eligible)
                prob_arr += (
                    np.mean(value_eligible)
                    * 0.001
                    * np.random.rand(value_eligible.shape[0])
                )
                prob_arr = prob_arr / np.sum(
                    prob_arr
                )  # normalize so they sum to 1

            next_loc_1d = np.random.choice(eligible_states_id, p=prob_arr)

        else:
            next_loc_1d = eligible_states_id[
                np.argmax(value_eligible)
            ]  # DETERMINISTIC POLICY that works
        return next_loc_1d 