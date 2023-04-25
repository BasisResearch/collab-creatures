# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:52:17 2023

@author: admin
"""

import numpy as np
import gridworld_utils as util

class Simulation(object):
    def __init__(self): 
        return


class Environment(object):

    def __init__(self, edge_size=30, N_total_food_units=16, patch_dim=1, max_step_size=3):

        self.edge_size = edge_size
        self.N_states = edge_size ** 2
        self.N_total_food_units = N_total_food_units
        self.patch_dim = patch_dim 
        self.food_decay_rate = 0.3 

        self.max_step_size = max_step_size 

        self.x_arr, self.y_arr, self.locs_1d_arr = util.create_2Dgrid(edge_size)
        self.T_prob, self.T_eligible = self.build_transition_matrix(max_step_size=self.max_step_size)

        self.phi_food = np.zeros([self.N_states, 1]) # indicator vector showing which locations are occupied by food. 
        self.food_calories_by_loc = np.zeros([self.N_states, 1]) # amount of food at each location in units of calories 
 
        return
    
    def build_transition_matrix(self, max_step_size=1):
        # This function is called when the environment is initialized and the transition matrix becomes fixed. 
        # The maximum step size is passed in as a parameter in the initialization function. 
        # After initialization you can still call this function and pass in an arbitrary value for the max step size to 
        # visualize what different transition matrices look like, but it won't change the transition matrix of the environment. 

        # compute eligible state transitions with a Euclidean distance rule 
        # (up, down, left ,right)

        T = np.zeros([self.N_states, self.N_states])

        for i in range(self.N_states):
            for j in range(self.N_states):
                T[i,j] = ( np.sqrt( (self.x_arr[j] - self.x_arr[i])**2 + (self.y_arr[j] - self.y_arr[i])**2 ) ) <= max_step_size # make this bigger to include more eligible states!!! 

        T_eligible = T # save the binary representation 
        T_prob = T / np.sum(T, axis=0, keepdims=True) 

        return T_prob, T_eligible
    
    # def set_transition_matrix(self, T_prob, T_eligible):
    #     self.T_prob = T_prob
    #     self.T_eligible = T_eligible
    #     return
    
    def add_food_patches(self, food_statistics_type="drop_food_once"):
        # returns the x and y locations of the new food locations 
        N_units_per_patch = self.patch_dim ** 2
        N_patches = np.ceil(self.N_total_food_units / N_units_per_patch).astype(int)

        if food_statistics_type == "drop_food_once":
            
            for pi in range(N_patches): 

                x_start = np.random.randint(0, self.edge_size - self.patch_dim)
                y_start = np.random.randint(0, self.edge_size - self.patch_dim)
                # generate (x,y) coordinates for each food unit in the patch 
                x_range, y_range = np.arange(x_start, x_start + self.patch_dim), np.arange(y_start, y_start + self.patch_dim)
                x_locs, y_locs = np.meshgrid(x_range, y_range, indexing='xy') 
                # convert to 1D locations 
                list_newfood_loc_1d = util.loc2Dto1D(x_locs.flatten(), y_locs.flatten(), self.edge_size)
                
                # update food tracking variables 
                self.phi_food[list_newfood_loc_1d] = 1  # boolean
                self.food_calories_by_loc[list_newfood_loc_1d] = 20 # add a fixed number of calories to each new food location 

        return 
    
    # def initialize_bird_agents(self, N_agents, agent_params):
    #     list_agents = []
    #     loc_1d_allagents = np.zeros(N_agents, dtype=int) # array containing location of each agent (index is agent ID)
    #     phi_agents = np.zeros([env.N_states, 1]) # # one-hot vector indicating how many agents are in each location (index is loc ID)
    
    #     for ai in range(N_agents):
    #         new_agent = BirdAgent(agent_params, self, N_timesteps)
    #         list_agents.append(new_agent)
            
    #         current_loc_id = np.random.randint(N_states) # pick a random location for eaceh agent
    #         new_agent.state_trajectory[0] = current_loc_id
            
    #         # update which locations are occupied by agents 
    #         loc_1d_allagents[ai] = current_loc_id   # list
    #         phi_agents[current_loc_id] += 1                    # add an agent to this location

    #     return list_agents, loc_1d_allagents, phi_agents

    
class BirdAgent(object):
    def __init__(self, env, N_timesteps, discount_factor, energy_init=50, sight_radius=40): 
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
        

        return 
    
    # def step_agent()
    # def update_calories()
    # 
        
    
    

class SimpleAgent(object): 
    def __init__(self, T_prob, N_states, N_timesteps, discount_factor, energy_init=50, sight_radius=40): 
        self.discount_factor = discount_factor # scalar between 0 and 1 
        self.SR = np.linalg.pinv(np.eye(N_states) - discount_factor * T_prob) # (N_states, N_states) matrix 
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([N_states, N_timesteps])
        self.phi_neighbors = np.zeros([N_states, 1]) # indicator  vector showing which locations do I see a neighboring agent?
        self.energy_total = energy_init # scalar quantity (calories)
        self.energy_trajectory = np.zeros([N_timesteps]) # not sure if we need this to be inside the class
        self.energy_trajectory[0] = energy_init
        self.times_at_food = [] # list of frames the agent is at a food location 
        self.sight_radius = sight_radius
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