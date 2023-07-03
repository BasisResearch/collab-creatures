# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:10:36 2023

building off of multiagent_foraging_mvp.py

@author: admin
"""

import TreeWorld
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
from pathlib import Path
import pickle
import figures 
import gridworld_utils as util
from importlib import reload
import time
reload(util)
reload(figures)

# ------------ Would you like to save the simulation? --------------------
saveData = True
directory_data = 'simulated_data/' 
filename_data = 'model_v2_distr_ignorers_patchdim1_self0p9_others0p1'

doAnimation = False
saveMovie = False
directory_movie = 'movies/'
filename_movie = 'new_movie'

# Turn on/off plots an animations
plot_initial_world = False
plot_T = False
plotValueFuncAtTime = False
do_plot_dist_to_neighbors = False
do_plot_numfailed = False
do_plot_timetofood = False
do_plot_calories =  False
do_plot_visible_food = False
do_plot_value_otheragents = False
do_plot_model_internals = False
doPrintAgentStateTrajectories = False


figures.setup_fig()
plt.close('all')

# ---------------------- Simulation parameters ------------------------------
N_sims = 100
N_timesteps = 50
N_agents = 9

# Food and environment parameters 
food_statistics_types = ["drop_food_once", "replenish_only_after_depleted", "regular_intervals", "poisson"] 
food_statistics_type = "drop_food_once"
N_food_units_total = 16
patch_dim = 4  #a patch has dimensions (patch_dim x patch_dim )
N_units_per_patch = patch_dim ** 2
N_patches = np.ceil(N_food_units_total / N_units_per_patch).astype(int)
calories_acquired_per_unit_time = 5 # when an agent is at a food location, it gains this many calories per time step 
epoch_dur = N_timesteps # add new food in random locations every epoch_dur time steps

# Agent parameters 
# doShareFoodInfo = True # Binary variable - are the birds communicating or not?
max_step_size = 3
sight_radius = 5
energy_init = 50
discount_factor = 0.9
c_food_self = 0.9
c_food_others = 0.1 # to what extent do the birds care about information from other birds?
c_otheragents = 0
c_group = 0
# c_predators = 0
c_weights = [c_food_self, c_food_others, c_otheragents, c_group]
caloric_cost_per_unit_dist = 1
doProbabilisticPolicy = True
doSoftmaxPolicy = True
exploration_bias = 0.001

# Quantities to track 
agent_locs_1d_allsims = np.zeros([N_sims, N_agents, N_timesteps])
dist_to_nearest_neighbor_allsims = np.zeros([N_sims, N_agents, N_timesteps])
calories_acquired_allsims = np.zeros([N_sims, N_agents, N_timesteps])
time_to_first_food_allsims = np.zeros([N_sims, N_agents])

# ************** CREATE ENVIRONMENT ***************************
edge_size = 30 # grid world has dimensions edge_size x edge_size
x_arr, y_arr, loc_1d_arr = util.create_2Dgrid(edge_size)

# --------------- Build transition matrix-----------------------------

N_states = edge_size ** 2
T = np.zeros([N_states, N_states])

# compute eligible state transitions with a Euclidean distance rule 
# (up, down, left ,right)
for i in range(N_states):
  for j in range(N_states):
    T[i,j] = ( np.sqrt( (x_arr[j] - x_arr[i])**2 + (y_arr[j] - y_arr[i])**2 ) ) <= max_step_size # make this bigger to include more eligible states!!! 

T_eligible = T # save the binary representation 
T_prob = T / np.sum(T, axis=0, keepdims=True) # normalization so elements represent probabilities 
 
if plot_T: 
    plt.figure()
    plt.title('Eligible state transitions')
    plt.imshow(T_eligible)
    
    plt.figure()
    plt.title('State transition probabilities')
    plt.imshow(T)
    

#----------------------------------------------------------------------------
# consider making a Simulation class and saving data from each simulation inside the object 
# or creating a pandas data frame 
starttime = time.time()

for si in range(N_sims):
    
    
    # ---------------------- Add food rewards -----------------------------------
    # TO DO: design a model of food dynamics 
    
    # Data structures to consider: 
    # 1) N_states x 1 one-hot vector indicating locations containing rewards
    # 2) N_states x 1 vector with magnitude of reward at each state/locations
    # 3) array or list of loc_ids (locations) containing rewards 
    
    phi_food = np.zeros([N_states, 1]) # indicator vector showing which locations are occupied by food. 
    food_calories_by_loc = np.zeros([N_states, 1]) # amount of food at each location in units of calories 
    food_trajectory = np.zeros([N_states, N_timesteps]) # track food calories over time 
    # list_food_loc_id = np.zeros([N_food]) # array of locations where there is food
    # list_food_loc_id = np.random.permutation(np.arange(N_states))[:N_food] # randomly choose K locations to place new food 
    # phi_food[list_food_loc_id] = 1 # put one food item in the respective locations
    food_init_loc_2d = np.reshape(phi_food, [edge_size,edge_size])
    
    if food_statistics_type == "drop_food_once":
        for pi in range(N_patches): 
            x_start = np.random.randint(0, edge_size - patch_dim)
            y_start = np.random.randint(0, edge_size - patch_dim) 
            # generate (x,y) coordinates for each food unit in the patch 
            x_range, y_range = np.arange(x_start, x_start + patch_dim), np.arange(y_start, y_start + patch_dim)
            x_locs, y_locs = np.meshgrid(x_range, y_range, indexing='xy') 
            # convert to 1D locations 
            list_newfood_loc_1d = util.loc2Dto1D(x_locs.flatten(), y_locs.flatten(), edge_size)
            
            # update food tracking variables 
            phi_food[list_newfood_loc_1d] = 1
            food_calories_by_loc[list_newfood_loc_1d] = 20 # add a fixed number of calories to each new food location 

    # if food_statistics_type == 'sequential': # new food appears only when old food is completely depleted 
        
    
    N_predators = 0
    w_predators = np.zeros([N_states, 1])  # vector indicating location of predators 
    list_predator_loc_id = np.random.permutation(np.arange(N_states))[:N_predators] # randomly choose K locations to place new food 
    w_predators[list_predator_loc_id] = 1
    predator_2d_locs = np.reshape(w_predators, [edge_size,edge_size])
    
    if plot_initial_world:
        fig_env, ax_env = plt.subplots()
        plt.imshow(food_init_loc_2d)
        # plt.imshow(food_2d_locs + predator_2d_locs)
    
    # food_loc_id = loc_1d_arr[np.random.randint(edge_size**2)] # pick a random loc_id
    # food_loc_id = edge_size**2 - 1 #put the food as far out as possible
    
    # for ai in range(N_food):
    #     x, y = util.loc1Dto2D(list_food_loc_id[ai], edge_size)
    #     ax_env.plot(x, y, 'ro')
    
    
    # ----------------------- Add agents -----------------------------------
    
    # *** Normalization of attention weights *** 
    # Normalize the magnitude of the attention weights so that the total magnitude sums to 1.
    # This ensures that each population or species has the same total amount of attention 
    # or reward capacity to allocate across different features of the environment. 
    c_weights = c_weights / np.sum(np.abs(c_weights))
    
    list_agents = []
    loc_1d_allagents = np.zeros(N_agents, dtype=int) # array containing location of each agent (index is agent ID)
    phi_agents = np.zeros([N_states, 1]) # # one-hot vector indicating how many agents are in each location (index is loc ID)
    
    # matrix tracking energy acquisition over time, used for determining fitness of the species
    calories_acquired_mat = np.zeros([N_agents, N_timesteps]) 
    calories_expended_mat = np.zeros([N_agents, N_timesteps]) 
    calories_total_mat = np.zeros([N_agents, N_timesteps]) 
    calories_cumulative_vec = np.zeros([N_agents, N_timesteps])
    
    # initialize the agents
    for ai in range(N_agents):
        new_agent = TreeWorld.SimpleAgent(T_prob, N_states, N_timesteps=N_timesteps, 
                                          discount_factor=discount_factor, 
                                          energy_init=energy_init, sight_radius=sight_radius)
        list_agents.append(new_agent)
        
        current_loc_id = np.random.randint(N_states) # pick a random location]
        new_agent.state_trajectory[0] = current_loc_id
        
        # update which locations are occupied by agents 
        loc_1d_allagents[ai] = current_loc_id   # list
        phi_agents[current_loc_id] += 1                    # add an agent to this location
        
        # agent.energy_trajectory[0] = 50 # each agent starts with 50 calories --> done inside the class
        
    calories_total_mat[:, 0] = energy_init # each agent starts with 50 calories
    
    # ************************** RUN SIMULATION  ***************************
    
    for ti in range(N_timesteps-1):
        print('sim ' + str(si) + ', time step ' + str(ti))
        
        ## ---------------------Update environment----------------------------
        # features = phi_food# (N_features, N_states) matrix, each row being 
        
        #Update agent calorie levels 
        # food occupied by an agent decays over time 
        # delta_food_calories_total = food_depletion_rate * food_calories_by_loc * phi_agents # only subtract food calories in locations occupied by agents, scaled by the number of agents 
        delta_food_calories_total =  calories_acquired_per_unit_time * phi_agents
        # rectify the calorie count for the food locations that will hit negative calories 
        is_overdepleted = delta_food_calories_total > food_calories_by_loc # find locations where the calorie count will hit negative values (we'll set the calorie count to 0)
        delta_food_calories_total[is_overdepleted] = food_calories_by_loc[is_overdepleted]

        food_calories_by_loc -= delta_food_calories_total
        phi_food = food_calories_by_loc  > 0.01 # update indicator  vector for food locations 
         
        # if food_statistics_type == "replenish_after_depletion":
        #     if np.sum(food_calories_by_loc) <= 0:
                
        if food_statistics_type == "regular_intervals":
            # randomly add a new food patch every several time steps  
            if ti % epoch_dur == 0:
                list_newfood_loc_id = np.random.permutation(np.arange(N_states))[:N_patches]
                phi_food[list_newfood_loc_id] = 1
                food_calories_by_loc[list_newfood_loc_id] = 20 # add a fixed number of calories to each new food location 
                # TO DO: make phi_food a calorie count (randomly pick between a range of calories) 
                
        # save food trajectory for plotting - how much food is in each location at each time step? 
        food_trajectory[:, ti+1] = food_calories_by_loc.flatten()    # (N_states, N_timesteps)   # save as a sparse matrix?
        
        
        #update predator locations
        
        ## ---------------------Update agents ---------------------------------
        
        for ai, agent in enumerate(list_agents):
            # sum_weighted_features = agent.c.T @ features 
            prev_loc_1d = int(agent.state_trajectory[ti]) # agent's current location 
            
            #------ update energy consequences of previous time step's actions --------
            
            #update agent's total energy based on amount of food at previous location  
            #transfer calories from food to agent
            calories_acquired_mat[ai, ti] = delta_food_calories_total[prev_loc_1d] / phi_agents[prev_loc_1d][0] # if there were N agents at that location, it gets 1/N portion of the calories
            agent.energy_total += calories_acquired_mat[ai, ti] 
            calories_cumulative_vec[ai, ti+1] = calories_cumulative_vec[ai, ti] + calories_acquired_mat[ai, ti] # only tracks calories acquired?
            
            # # remove this agent from the list of surviving agents if it's energy reaches zero 
            # if agent.energy_total <= 0:
            #     list_deceased_agents = list_surviving_agents.pop(ai)  # be careful b/c the rest of this for loop assumes all the agents are alive
            
            # -------------- Compute expected rewards, values, and make a decision --------------------------------  
            
            phi_agents[prev_loc_1d] -= 1 # move out of previous location
            
            # EXPECTED REWARD RELATED TO OTHER AGENTS
            xloc_allagents, yloc_allagents = util.loc1Dto2D(loc_1d_allagents, edge_size)
            xloc_self, yloc_self = util.loc1Dto2D(prev_loc_1d, edge_size)
            # only include locations of agents outside of current location
            xloc_neighbors, yloc_neighbors = util.loc1Dto2D(loc_1d_allagents[loc_1d_allagents != prev_loc_1d], edge_size)
            
            # expected reward at each location based on proximity to other agents 
            w_otheragents_2d = agent.reward_function_otheragents(xloc_neighbors, yloc_neighbors, xloc_self, yloc_self, edge_size)
            w_otheragents_1d = np.reshape(w_otheragents_2d, (N_states, 1))
            
            # EXPECTED REWARD RELATED TO CENTER OF MASS
            xloc_otheragents = np.delete(xloc_allagents, ai)  # remove this agent's own location from the list 
            yloc_otheragents = np.delete(yloc_allagents, ai) # 
            if N_agents > 1:
                xloc_centerofmass, yloc_centerofmass = util.center_of_mass(xloc_otheragents, yloc_otheragents)
            else:
                xloc_centerofmass, yloc_centerofmass = xloc_self, yloc_self
                 
            # expected reward of each location based on this agent's distance from center of mass of the group
            w_groupcenterofmass = np.zeros([edge_size, edge_size])
            w_groupcenterofmass[int(yloc_centerofmass), int(xloc_centerofmass)] = 0.5 
            w_groupcenterofmass = np.reshape(w_groupcenterofmass, (N_states, 1))
            
            # VISIBILITY CONSTRAINTS
            phi_visible_mat = agent.compute_visible_locations(xloc_self, yloc_self, edge_size)
            phi_visible = np.reshape(phi_visible_mat, (N_states, 1))
            
            # get information from other agents about whether there is food at their locations 
            # if doShareFoodInfo:
            #     phi_visible[loc_1d_allagents] = 1   # can this agent see the locations of other agents?
                # It's not quite communication between agents yet because there is no 
                # capacity for misrepresentation here - the agent simply has information about other agent's locations.
                # The info from other agents should be represented separately from the agent's own information. 
                # Then you can change communication parameters such as fidelity of info transmitted (add noise) 
            
            # EXPECTED REWARD RELATED TO FOOD
            w_food = phi_food * phi_visible # expected food reward at each location
            w_food_others = phi_food * phi_agents  # making food info from other agents a separate feature with separate weights
            
            sum_weighted_features = \
                  c_weights[0] * w_food \
                + c_weights[1] * w_food_others \
                + c_weights[2] * w_otheragents_1d   \
                + c_weights[3] * w_groupcenterofmass 
                        
            # sum_weighted_features = c_food_self * phi_food  + c_predator * phi_predator #+ c_otheragents * w_otheragents_1d
            
            # VALUE FUNCITON 
            value = agent.SR @ sum_weighted_features         # (N_states, 1) vector
            
            # POLICY: select next action using the value and eligible states 
            # eligible states are those specified by the transition matrix. Can constrain further to exclude states not occupied by other agents
            eligible_states_id = np.nonzero(T_eligible[:, prev_loc_1d])[0]  # * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
            value_eligible = value[eligible_states_id].flatten()   # value of eligible states plus some noise 
            
            # ACTION SELECTION
            
            if doProbabilisticPolicy:
                if doSoftmaxPolicy:
                    prob_arr = util.softmax(value_eligible, temp=exploration_bias)
                else:
                
                    # #sample eligible states from a categorical distribution whose shape is based on the values 
                    # # convert values into probabilities
                    # value_eligible += 0.001 * np.random.randn(value_eligible.shape[0]) # add some noise 
                    # prob_arr = value_eligible - np.min(value_eligible) # shift values so they are all positive and add some noise
                    prob_arr = value_eligible - np.min(value_eligible)
                    prob_arr += np.mean(value_eligible) * 0.001 * np.random.rand(value_eligible.shape[0])
                    prob_arr = prob_arr / np.sum(prob_arr) # normalize so they sum to 1
                
                next_loc_1d = np.random.choice(eligible_states_id, p=prob_arr) 
    
                
            else:
                next_loc_1d = eligible_states_id[np.argmax(value_eligible)]  # DETERMINISTIC POLICY that works
                
                
            # ------- compute energy cost of moving to new location -------------- 
            
            xloc_prev, yloc_prev = util.loc1Dto2D(prev_loc_1d, edge_size)
            xloc_next, yloc_next = util.loc1Dto2D(next_loc_1d, edge_size)
            dist_traveled = np.sqrt((xloc_next - xloc_prev)**2 + (yloc_next - yloc_prev)**2)
            calories_expended_mat[ai,ti] = caloric_cost_per_unit_dist * dist_traveled
            agent.energy_total -= calories_expended_mat[ai,ti]
            
            #------------- compute metrics for data analysis -----------------
            if len(list_agents) > 1:
                dist_to_neighbors = np.sqrt((xloc_otheragents - xloc_self) ** 2 + (yloc_otheragents - yloc_self) ** 2 )
                dist_to_nearest_neighbor_allsims[si, ai, ti] = np.min(dist_to_neighbors) 
            
            calories_acquired_allsims[si, ai, ti] = calories_acquired_mat[ai, ti] 
            
            if phi_food[next_loc_1d][0]:
                agent.times_at_food.append(ti+1) # add this frame to the list of frames where agent is at a food location
            
            # ------------------------------------------------------------------- 
            
            agent.state_trajectory[ti+1] = next_loc_1d        # scalar 
            agent.value_trajectory[:, ti+1] = value.flatten()          # (N_states, N_timesteps)   
            agent.energy_trajectory[ti+1] = agent.energy_total   
            calories_total_mat[ai, ti+1] = calories_total_mat[ai, ti] + calories_acquired_mat[ai, ti] - calories_expended_mat[ai,ti]
            
            phi_agents[next_loc_1d] += 1                     # move into new location 
            loc_1d_allagents[ai] = next_loc_1d
    

    for ai, agent in enumerate(list_agents):
        # print(agent.times_at_food)
        if len(agent.times_at_food) == 0:
            time_to_first_food_allsims[si, ai] = N_timesteps # TO DO: figure out how to track agents who never reach food
        else:
            time_to_first_food_allsims[si, ai] = agent.times_at_food[0] # fetch the first time step where agent is at a food location
    
endtime = time.time()

if doPrintAgentStateTrajectories:
    for ai, agent in enumerate(list_agents):
        print(agent.state_trajectory)
    
print('simulation run time = ' + str(endtime - starttime))

#%% Quantify foraging statistics after running the sims

#%% Number of birds that reach food within the duration of the simulation
did_reach_food = time_to_first_food_allsims < N_timesteps  # did this bird reach food in time? (N_sims, N_agents)
num_agents_did_reach_food = np.sum(did_reach_food, axis=1) # number of birds that reached food in time (N_sims, 1)

# number of agents that failed to reach food 
failed_to_reach_food = 1 - did_reach_food  # did this bird reach food in time? (N_sims, N_agents)
num_agents_failed_reach_food = np.sum(failed_to_reach_food, axis=1) # number of birds that reached food in time (N_sims, 1)
 
if do_plot_numfailed:
    fig, ax = plt.subplots()
    ax.hist(num_agents_failed_reach_food, bins=N_agents+1)
    # ax.set_title('Number of birds that failed to reach food')
    ax.set_xlabel('$N_{failed}$')
    ax.set_ylabel('Number of individuals')
    ax.set_xlim([0, N_agents + 1 ])
    fig.tight_layout()

#%% Time to first food item

pop_mean_time_to_first_food = np.mean(time_to_first_food_allsims, axis=1) # (N_sims,) mean across individuals in the population
pop_var_time_to_first_food = np.var(time_to_first_food_allsims, axis=1)

if do_plot_timetofood:

    # Distribution over populations 
    fig, ax = plt.subplots()
    ax.hist(pop_mean_time_to_first_food, bins=np.arange(N_timesteps + 2))
    ax.set_xlabel('Time to food location')
    ax.set_ylabel('Number of populations')
    ax.set_xlim([0, N_timesteps+2 ])
    ax.set_xticks(np.linspace(0, N_timesteps, 6).astype(int))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(pop_var_time_to_first_food, bins=np.arange(N_timesteps))
    ax.set_title('Time to food location \n (variance across population)')
    ax.set_ylabel('Number of populations')
    ax.set_xlim([0, N_timesteps+2 ])
    fig.tight_layout()
    
    #Distribution over individuals for the very last population simulated
    si = 0
    fig, ax = plt.subplots()
    ax.hist(time_to_first_food_allsims[si], bins=np.arange(N_timesteps+2))
    ax.set_xlabel('Time to food location')
    ax.set_ylabel('Number of individuals')
    ax.set_xlim([0, N_timesteps +2 ])
    fig.tight_layout()


#%%  
dist_to_nearest_neighbor_time_mean = np.mean(dist_to_nearest_neighbor_allsims, axis=2)
dist_to_nearest_neighbor_pop_mean = np.mean(dist_to_nearest_neighbor_allsims, axis=(1,2))
dist_to_nearest_neighbor_pop_var = np.var(dist_to_nearest_neighbor_allsims, axis=(1,2))

calories_acquired_per_unit_time = np.sum(calories_acquired_allsims, axis=(2)) / N_timesteps
calories_acquired_pop_mean = np.mean(calories_acquired_per_unit_time, axis=(1))
calories_acquired_pop_var = np.var(calories_acquired_per_unit_time, axis=(1))


# # Distribution over population
# si = 0 # sim index 
# fig, ax = plt.subplots()
# ax.hist(dist_to_nearest_neighbor_time_mean[0, :])
# ax.set_title('Mean distance to nearest neighbor (across population)')
# fig.tight_layout()


if do_plot_dist_to_neighbors:
    # Distribution over simulations 
    fig, ax = plt.subplots()
    ax.hist(dist_to_nearest_neighbor_pop_mean)
    ax.set_title('Mean distance to nearest neighbor (across population and time)')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(dist_to_nearest_neighbor_pop_mean)
    ax.set_title('Variance distance to nearest neighbor (across population and time)')
    fig.tight_layout()

if do_plot_calories:
    # Distribution over simulations 
    fig, ax = plt.subplots()
    ax.hist(calories_acquired_pop_mean)
    ax.set_title('Mean calories acquired per unit time (across population)')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(calories_acquired_pop_var)
    ax.set_title('Variance calories acquired per unit time (across population)')
    fig.tight_layout()



#%% Quantify foraging success or evolutionary fitness

if do_plot_calories:

    # plot each agent's cumulative calorie acquitision over time 
    # ai = 0
    fig, ax = plt.subplots()
    for ai in range(N_agents):
        # ax.plot(list_agents[ai].energy_trajectory, '.-')
        ax.plot(calories_cumulative_vec[ai,:], '.-')
    # ax.set_title('Agent ' + str(ai))
    ax.set_xlabel('Time step')
    ax.set_ylabel('Total calories acquired')
    # ax.set_ylim([0, np.max(food_trajectory)])
    fig.tight_layout()
    
    # plot each agent's energy level over time 
    # ai = 0
    fig, ax = plt.subplots()
    for ai in range(N_agents):
        # ax.plot(list_agents[ai].energy_trajectory, '.-')
        ax.plot(calories_total_mat[ai, :], '.-')
    # ax.set_title('Agent ' + str(ai))
    ax.set_xlabel('Time step')
    ax.set_ylabel('Total calories')
    fig.tight_layout()
    
    # Population-level distribution of average calories acquired over time
    mean_calories_acquired_time = np.mean(calories_acquired_mat, axis=1)
    fig, ax = plt.subplots()
    ax.hist(mean_calories_acquired_time)
    ax.set_xlabel('Mean calories acquired (across time)')
    fig.tight_layout()
    
#%% ***************  PLOTS OF MODEL INTERNALS *********************************

if do_plot_model_internals: 
    #%% State (phi_agent)
    phi_agent = np.zeros([N_states, 1])
    phi_agent[next_loc_1d] = 1 
    phi_agent_2d = np.reshape(phi_agent, (edge_size, edge_size))
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=phi_agent_2d, yticklabels=False, xticklabels=False, 
                square=True, vmin=0, vmax=1, cbar=False, cmap=sns.color_palette("mako", as_cmap=True))
    ax.set_title('Agent state')
    
    #%% Successor Representation Matrix applied to state (M @ phi_agent) 
    SR_phi = agent.SR @ phi_agent
    SR_phi_2d = np.reshape(SR_phi, (edge_size, edge_size))
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=SR_phi_2d, yticklabels=False, xticklabels=False, 
                square=True, cbar=True, norm=LogNorm(), cmap=sns.color_palette("mako", as_cmap=True))
    ax.set_title('Successor Representation applied to agent state')
    
    # log_SR_phi_2d = np.log(SR_phi_2d)
    # vmin = np.min(log_SR_phi_2d)
    # vmax = np.max(log_SR_phi_2d)
    # fig, ax = plt.subplots()
    # sns.heatmap(ax=ax, data=log_SR_phi_2d, yticklabels=False, xticklabels=False, 
    #             square=True, cbar=True, vmin=vmin, vmax=vmax, cmap=sns.color_palette("mako", as_cmap=True))
    
    #%% Expected reward computed for all locations ( w.T @  M )
    
    value_food = w_food.T @ agent.SR
    value_food_2d = np.reshape(value_food, (edge_size, edge_size))
    
    value_otheragents = w_otheragents_1d.T @ agent.SR
    value_otheragents_2d = np.reshape(value_otheragents, (edge_size, edge_size))
    
    # plots
    
    vmin = np.min(value_food_2d)
    vmax = np.max(value_food_2d)
    if vmin <= 0: 
        vmin = 1e-5
    if vmax <= 0:
        vmax = 1
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=value_food_2d, yticklabels=False, xticklabels=False, 
                square=True, cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=sns.color_palette("rocket", as_cmap=True))
                # square=True, vmin=0, vmax=1,  cbar=True,  cmap=sns.color_palette("rocket", as_cmap=True))
    # ax.set_title('$ w_{food}^T M$')
    
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=value_otheragents_2d , yticklabels=False, xticklabels=False, 
                square=True,   cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=sns.color_palette("rocket", as_cmap=True))
                # square=True, vmin=0, vmax=1,  cbar=True, norm=LogNorm(), cmap=sns.color_palette("rocket", as_cmap=True))
    # ax.set_title('$ w_{neighbors}^T M $')
    
    #%% Weighted value function
    
    value_food = c_food_self * w_food.T @ agent.SR
    value_food_2d = np.reshape(value_food, (edge_size, edge_size))
    
    value_otheragents = c_otheragents * w_otheragents_1d.T @ agent.SR
    value_otheragents_2d = np.reshape(value_otheragents, (edge_size, edge_size))
    
    sum_weighted_vals_2d =  value_food_2d + value_otheragents_2d
    val_weighted_sum = agent.SR @ ( c_food_self * w_food  + c_otheragents * w_otheragents_1d)
    val_weighted_sum_2d = np.reshape( val_weighted_sum , (edge_size, edge_size))
    
    vmax = np.max(value_food_2d)
    if vmin <= 0: 
        vmin = 1e-5
    if vmax <= 0:
        vmax = 1
    
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=value_food_2d, yticklabels=False, xticklabels=False, 
                square=True, cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax),  cmap=sns.color_palette("rocket", as_cmap=True))
    # ax.set_title('$c_{food} w_{food}^T M$')
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=value_otheragents_2d , yticklabels=False, xticklabels=False, 
                square=True,  cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=sns.color_palette("rocket", as_cmap=True))
    # ax.set_title('$c_{neighbors} w_{neighbors}^T M$')
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=sum_weighted_vals_2d , yticklabels=False, xticklabels=False, 
                square=True,  cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=sns.color_palette("rocket", as_cmap=True))
    # ax.set_title('$V(S)$')
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=sum_weighted_vals_2d , yticklabels=False, xticklabels=False, 
                square=True,  cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=sns.color_palette("rocket", as_cmap=True))
    ax.set_title('M (c_{food}w_{food}^T + c_{neighbors} w_{neighbors}^T )')
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=np.reshape(value, (edge_size, edge_size)), yticklabels=False, xticklabels=False, 
                square=True, cbar=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=sns.color_palette("rocket", as_cmap=True))
    ax.set_title('from sim')
    
    
    #%% Reward locations (phi_food)
    w_2d = np.reshape(w_food, (edge_size, edge_size))
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=w_2d, yticklabels=False, xticklabels=False, 
                square=True, vmin=0, vmax=1, cbar=False, cmap=sns.color_palette("rocket", as_cmap=True))

#%% Value ( w.T @  M @ phi_agent )

# w_SR = w_food.T @ agent.SR
# w_SR_2d = np.reshape(w_SR, (edge_size, edge_size))

# fig, ax = plt.subplots()
# sns.heatmap(ax=ax, data=w_SR_2d, yticklabels=False, xticklabels=False, 
#             square=True, vmin=0, vmax=1, center=0, cbar=False)


#%% plot food reward map for one agent

if do_plot_visible_food: 
    f_food_2d = np.reshape(phi_food * phi_visible, (edge_size, edge_size))
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=f_food_2d, vmin=0, vmax=1, center=0, cbar=True)
    ax.plot(xloc_self, yloc_self, '.')
    ax.set_title('perceived food location')
    
    
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=phi_visible_mat, vmin=0, vmax=1, center=0, cbar=True)
    ax.plot(xloc_self, yloc_self, '.')
    ax.set_title('Sight radius for food')

#%% plot social reward map for one agent

if do_plot_value_otheragents: 
    fig, ax = plt.subplots()
    sns.heatmap(ax=ax, data=w_otheragents_2d, center=0, cbar=True)
    ax.set_title('social reward map')

#%% plot probability of choosing a location as a function of value of that location

if doProbabilisticPolicy:
    fig, ax = plt.subplots(); 
    ax.bar(value_eligible, prob_arr, width=0.001); 
    fig.tight_layout()
    
    fig, ax = plt.subplots(); 
    ax.plot(value_eligible, prob_arr, '.'); 
    ax.set_ylim([0,1])
    fig.tight_layout()
    
    fig, ax = plt.subplots(); 
    ax.hist(value_eligible); 
    ax.set_title('distribution of values for eligible locations')
    fig.tight_layout()
    
    fig, ax = plt.subplots(); 
    ax.hist(prob_arr); 
    ax.set_title('distribution of probabilities')
    fig.tight_layout()

#%%  #------- plot value function for all agents at a specific time point -------

if plotValueFuncAtTime: 
    ti = 1 # which timestep to look at
    
    ncols = 3
    nrows = int(np.ceil(N_agents / ncols))
    grid_kws = {'wspace': 0.5}
    fig, ax = plt.subplots(nrows, ncols, gridspec_kw=grid_kws)
    plt.suptitle('Values at time ti = ' + str(ti))
    for  ai, agent in enumerate(list_agents):
        r = int(np.ceil((ai+1) / ncols) - 1)
        c = ai % ncols
        value_heatmap = np.reshape(list_agents[ai].value_trajectory[:, ti], [edge_size,edge_size]) 
        sns.heatmap(ax=ax[r,c], data=value_heatmap,  center=0, vmin=-1, vmax=1, square=True)
        ax[r,c].set_title('value for agent ' + str(ai))


#%%
# ************************* CREATE USER INTERFACE ***************************

if doAnimation: 
    
    fig_ui, (ax_main, ax_value) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax_main.set_title('Environment')
    # phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])
    min_cal = -0.1
    max_cal = np.max(food_trajectory)
    sns.heatmap(ax=ax_main, data=food_init_loc_2d, vmin=min_cal, vmax=max_cal, center=0, square=True, cbar=True)
    
    # Choose one agent to highlight in magenta. Plot in an adjacent 
    # panel the value function of this agent evolving over time 
    z = 0 # index of highlighted agent
    value_heatmap = np.reshape(list_agents[z].value_trajectory[:, 0], [edge_size,edge_size]) 
    #scale the color bar based on the c weights (TO DO: find a better way to do this )
    vmin = np.min(np.min(c_weights), 0)
    vmax = np.max(np.max(c_weights), 0)
    min_value = 0
    max_value = 1
    sns.heatmap(ax=ax_value, data=value_heatmap, vmin=min_value, vmax=max_value, center=0, square=True, cbar=True)
    ax_value.set_title('Value function for one agent')
    
    list_plot_agents = []
    markers = ['co'] # color for all other agents 
    
    # plot each agent's data
    for  ai, agent in enumerate(list_agents):
        if ai == z:
            marker = 'mo'
        else:
            marker = markers[0]
        one_plot, = ax_main.plot([], [], marker, markersize=3)   
        list_plot_agents.append(one_plot)
    
    
    
    def update_plot(frame, list_agents, food_trajectory, edge_size): #, list_food_loc_id, edge_size):
        fig_ui.suptitle('frame ' + str(frame) + ' / ' + str(N_timesteps))
        # food locations and quantities
        food_heatmap = np.reshape(food_trajectory[:, frame], [edge_size,edge_size]) 
        sns.heatmap(ax=ax_main, data=food_heatmap, vmin=min_cal, vmax=max_cal, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
        
        # x_food, y_food = util.loc1Dto2D(list_food_loc_id, edge_size)
        # plot_food.set_data(x_food, y_food)
        
        # state of each agent 
        for  ai, agent in enumerate(list_agents):
            state = int(agent.state_trajectory[frame])
            x, y = util.loc1Dto2D(state, edge_size)
            list_plot_agents[ai].set_data(x, y)
        
        # value function of one agent 
        ax_value.cla()
        value_heatmap = np.reshape(list_agents[z].value_trajectory[:, frame], [edge_size,edge_size]) 
        sns.heatmap(ax=ax_value, data=value_heatmap, vmin=min_value, vmax=max_value, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
        
        return list_plot_agents #, plot_food
    
    fig_ui.tight_layout()
    
    ani = animation.FuncAnimation(fig_ui, update_plot, frames=N_timesteps,
                                  fargs=(list_agents, food_trajectory, edge_size)) #, list_food_loc_id, edge_size))
    

#%% **********************  SAVE DATA ****************************************
# saveData = False

# filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\simulated_data\sim1"
filepath_data = directory_data + filename_data
file_data = Path(filepath_data + '.sav')
# saving to m4 using ffmpeg writer
if file_data.exists():
    filepath_data = filepath_data + '_v2'

# save the model to disk
# filename = filepath + '.sav'
dictionary = {'N_sims':N_sims, 'N_timesteps':N_timesteps, 'N_agents':N_agents, \
              'pop_mean_time_to_first_food': pop_mean_time_to_first_food, \
              'num_agents_failed_reach_food': num_agents_failed_reach_food, \
              'N_food_units_total':N_food_units_total, 'patch_dim':patch_dim, \
                  'example_agent':agent, 'sight_radius':sight_radius, 
                  'discount_factor':discount_factor,  'c_weights':c_weights}

pickle.dump(dictionary, open(filepath_data + '.sav', 'wb'))


#%% **********************  SAVE ANIMATION  ****************************************

# saveMovie = False

if saveMovie:
    # filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_foodonly_v2.gif"
    # filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_simple_explorebias10"
    
    filepath_movie = directory_movie + filename_movie
    moviefile = Path(filepath_movie + '.gif')
    # saving to m4 using ffmpeg writer
    if moviefile.exists():
        filepath_movie = filepath_movie + '_v2'

    ani.save(filepath_movie + '.gif', dpi=300, writer=animation.PillowWriter(fps=2))
    
    # ## saving as an mp4
    # filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_mp4.mp4"
    # ani.save(filename, writer=animation.FFMpegWriter(fps=30) )
 