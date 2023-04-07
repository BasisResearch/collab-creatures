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
from scipy import stats
import figures 
import gridworld_utils as util
from importlib import reload
import time
reload(util)
reload(figures)

figures.setup_fig()
plt.close('all')
plot_initial_world = False
plot_T = False
plotValueFuncAtTime = False
doPlotDistanceToNeighbor = False
doPlotCalories = False
doPrintAgentStateTrajectories = False

starttime = time.time()

# ************** CREATE ENVIRONMENT ***************************

edge_size = 30 # grid world has dimensions edge_size x edge_size
x_arr, y_arr, loc_id_arr = util.create_2Dgrid(edge_size)

# --------------- Build transition matrix-----------------------------

N_states = edge_size ** 2
T = np.zeros([N_states, N_states])

# compute eligible state transitions with a Euclidean distance rule 
# (up, down, left ,right)
for ai in range(N_states):
  for j in range(N_states):
    T[ai,j] = ( np.sqrt( (x_arr[j] - x_arr[ai])**2 + (y_arr[j] - y_arr[ai])**2 ) ) <= 3 # make this bigger to include more eligible states!!! 

T_eligible = T # save the binary representation 
T_prob = T / np.sum(T, axis=0, keepdims=True) # normalization so elements represent probabilities 
 
if plot_T: 
    plt.figure()
    plt.title('Eligible state transitions')
    plt.imshow(T_eligible)
    
    plt.figure()
    plt.title('State transition probabilities')
    plt.imshow(T)
    
    
# ---------------------- Simulation parameters ------------------------------
N_sims = 1
N_timesteps = 50
N_agents = 9

#food statistics
food_statistics_type = "static" #  also try "regular_intervals"
# food_statistics_type = "regular_intervals" #  also try "regular_intervals"
N_food_units_total = 3
patch_dim = 1 
N_units_per_patch = patch_dim ** 2
N_patches = np.ceil(N_food_units_total / N_units_per_patch).astype(int)
food_depletion_rate = 0 
epoch_dur = N_timesteps # add new food in random locations every epoch_dur time steps

# Quantities to track 
agent_locs_1d_allsims = np.zeros([N_sims, N_agents, N_timesteps])
dist_to_nearest_neighbor_allsims = np.zeros([N_sims, N_agents, N_timesteps])
calories_acquired_allsims = np.zeros([N_sims, N_agents, N_timesteps])
time_to_first_food_allsims = np.zeros([N_sims, N_agents])


#----------------------------------------------------------------------------
# consider making a Simulation class and saving data from each simulation inside the object 
# or creating a pandas data frame 
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
    
    if food_statistics_type == "static":
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
            # TO DO: make phi_food a calorie count (randomly pick between a range of calories) 
    
    
    N_predators = 3
    f_predators = np.zeros([N_states, 1])  # vector indicating location of predators 
    list_predator_loc_id = np.random.permutation(np.arange(N_states))[:N_predators] # randomly choose K locations to place new food 
    f_predators[list_predator_loc_id] = 1
    predator_2d_locs = np.reshape(f_predators, [edge_size,edge_size])
    
    if plot_initial_world:
        fig_env, ax_env = plt.subplots()
        plt.imshow(food_init_loc_2d)
        # plt.imshow(food_2d_locs + predator_2d_locs)
    
    # food_loc_id = loc_id_arr[np.random.randint(edge_size**2)] # pick a random loc_id
    # food_loc_id = edge_size**2 - 1 #put the food as far out as possible
    
    # for ai in range(N_food):
    #     x, y = util.loc1Dto2D(list_food_loc_id[ai], edge_size)
    #     ax_env.plot(x, y, 'ro')
    
    
    # ----------------------- Add agents -----------------------------------
    energy_init = 50
    discount_factor = 0.8
    c_food = 1
    c_predators = 0
    c_otheragents = 0
    c_group = 0
    c_weights = [c_food, c_otheragents, c_group]
    caloric_cost_per_unit_dist = 1
    doProbabilisticPolicy = False
    doSoftmaxPolicy = True
    exploration_bias = 0.01
    
    list_agents = []
    arr_loc_id_allagents = np.zeros(N_agents, dtype=int) # array containing location of each agent (index is agent ID)
    phi_agents = np.zeros([N_states, 1]) # # one-hot vector indicating which locations are occupied by agents (index is loc ID)
    
    # matrix tracking energy acquisition over time, used for determining fitness of the species
    calories_acquired_mat = np.zeros([N_agents, N_timesteps]) 
    calories_expended_mat = np.zeros([N_agents, N_timesteps]) 
    calories_total_mat = np.zeros([N_agents, N_timesteps]) 
    calories_cumulative_vec = np.zeros([N_agents, N_timesteps])
    
    # initialize the agents
    for ai in range(N_agents):
        new_agent = TreeWorld.SimpleAgent(T_prob, N_states, N_timesteps=N_timesteps, 
                                          discount_factor=discount_factor, energy_init=energy_init)
        list_agents.append(new_agent)
        
        current_loc_id = np.random.randint(N_states) # pick a random location]
        new_agent.state_trajectory[0] = current_loc_id
        
        # update which locations are occupied by agents 
        arr_loc_id_allagents[ai] = current_loc_id   # list
        phi_agents[current_loc_id] = 1                    # one-hot vector
        
        # agent.energy_trajectory[0] = 50 # each agent starts with 50 calories --> done inside the class
        
    calories_total_mat[:, 0] = energy_init # each agent starts with 50 calories
    
    # ************************** RUN SIMULATION  ***************************
    
    for ti in range(N_timesteps-1):
        print('sim ' + str(si) + ', time step ' + str(ti))
        
        ## ---------------------Update environment----------------------------
        # features = phi_food# (N_features, N_states) matrix, each row being 
        
        #Update food energy 
        # food occupied by an agent decays over time 
        delta_food_calories = food_depletion_rate * food_calories_by_loc    # change in magnitude of food amount at this time step 
        food_calories_by_loc -= delta_food_calories * phi_agents # only subtract food calories in locations occupied by agents
        phi_food = food_calories_by_loc  > 0.01 # update indicator  vector for food locations 
         
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
            agent.energy_total += delta_food_calories[prev_loc_1d]
            calories_acquired_mat[ai, ti] = delta_food_calories[prev_loc_1d]
            calories_cumulative_vec[ai, ti+1] = calories_cumulative_vec[ai, ti] + calories_acquired_mat[ai, ti]
            
            # # remove this agent from the list of surviving agents if it's energy reaches zero 
            # if agent.energy_total <= 0:
            #     list_deceased_agents = list_surviving_agents.pop(ai)  # be careful b/c the rest of this for loop assumes all the agents are alive
            
            # -------------- Make a decision --------------------------------  
            
            phi_agents[prev_loc_1d] = 0 # move out of previous location
            agent.phi_neighbors = phi_agents  # assume this agent knows the locations of all other agents
            
            # sum_weighted_features = c_food * phi_food + c_otheragents * agent.phi_neighbors
            
            # REWARD FUNCTION: compute weighted sum of features
            xloc_allagents, yloc_allagents = util.loc1Dto2D(arr_loc_id_allagents, edge_size)
            xloc_self, yloc_self = util.loc1Dto2D(prev_loc_1d, edge_size)
            xloc_otheragents = np.delete(xloc_allagents, ai)  # remove this agent's own location from the list 
            yloc_otheragents = np.delete(yloc_allagents, ai) # 
            f_otheragents_2d = agent.reward_function_otheragents(xloc_otheragents, yloc_otheragents, xloc_self, yloc_self, edge_size)
            f_otheragents_1d = np.reshape(f_otheragents_2d, (N_states, 1))
            if N_agents > 1:
                xloc_centerofmass, yloc_centerofmass = util.center_of_mass(xloc_otheragents, yloc_otheragents)
            else:
                xloc_centerofmass, yloc_centerofmass = xloc_self, yloc_self
            #compute distance from center of mass of each agent. 
            f_groupcenterofmass = np.zeros([edge_size, edge_size])
            f_groupcenterofmass[int(yloc_centerofmass), int(xloc_centerofmass)] = 0.5 
            f_groupcenterofmass = np.reshape(f_groupcenterofmass, (N_states, 1))
            
            phi_visible_mat = agent.compute_visible_locations(xloc_self, yloc_self, edge_size)
            phi_visible = np.reshape(phi_visible_mat, (N_states, 1))
            
            # sum_weighted_features = c_food * phi_food   + c_otheragents * f_otheragents_1d  
            sum_weighted_features = c_food * phi_food * phi_visible \
                + c_predators * f_predators \
                + c_otheragents * f_otheragents_1d   \
                + c_group * f_groupcenterofmass 
                        
            # sum_weighted_features = c_food * phi_food  + c_predator * phi_predator #+ c_otheragents * f_otheragents_1d
            
            # VALUE FUNCITON 
            value = agent.SR @ sum_weighted_features         # (N_states, 1) vector
            
            # POLICY: select next action using the value and eligible states 
            # eligible states are those specified by the transition matrix and states not occupied by other agents
            eligible_states_id = np.nonzero(T_eligible[:, prev_loc_1d] * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
            value_eligible = value[eligible_states_id].flatten()   # value of eligible states plus some noise 
            
            if doProbabilisticPolicy:
                if doSoftmaxPolicy:
                    prob_arr = util.softmax(value_eligible, T=exploration_bias)
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
            dist_to_neighbors = np.sqrt((xloc_otheragents - xloc_self) ** 2 + (yloc_otheragents - yloc_self) ** 2 )
            dist_to_nearest_neighbor_allsims[si, ai, ti] = np.min(dist_to_neighbors) 
            calories_acquired_allsims[si, ai, ti] = calories_acquired_mat[ai, ti] 
            
            if phi_food[prev_loc_1d][0]:
                agent.times_at_food.append(ti) # add this frame to the list of frames where agent is at a food location
            
            # ------------------------------------------------------------------- 
            
            agent.state_trajectory[ti+1] = next_loc_1d        # scalar 
            agent.value_trajectory[:, ti+1] = value.flatten()          # (N_states, N_timesteps)   
            agent.energy_trajectory[ti+1] = agent.energy_total   
            calories_total_mat[ai, ti+1] = calories_total_mat[ai, ti] + calories_acquired_mat[ai, ti] - calories_expended_mat[ai,ti]
            
            phi_agents[next_loc_1d] = 1                     # move into new location 
            arr_loc_id_allagents[ai] = next_loc_1d
    

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

#%% Time to first food item

pop_mean_time_to_first_food = np.mean(time_to_first_food_allsims, axis=0) # (N_sims,) mean across individuals in the population
pop_var_time_to_first_food = np.var(time_to_first_food_allsims, axis=0)

# Distribution over population for one sim 
si = 0
fig, ax = plt.subplots()
ax.hist(time_to_first_food_allsims[si])
ax.set_title('Time to first food item \n (distr. over individuals)')
fig.tight_layout()

# Distribution over simulations
fig, ax = plt.subplots()
ax.hist(pop_mean_time_to_first_food)
ax.set_title('Time to first food item \n (mean across population)')
fig.tight_layout()

fig, ax = plt.subplots()
ax.hist(pop_var_time_to_first_food)
ax.set_title('Time to first food item \n (variance across population)')
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

if doPlotDistanceToNeighbor:
    # Distribution over simulations 
    fig, ax = plt.subplots()
    ax.hist(dist_to_nearest_neighbor_pop_mean)
    ax.set_title('Mean distance to nearest neighbor (across population and time)')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(dist_to_nearest_neighbor_pop_mean)
    ax.set_title('Variance distance to nearest neighbor (across population and time)')
    fig.tight_layout()

if doPlotCalories:
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

if doPlotCalories:

    # plot each agent's cumulative calorie acquitision over time 
    # ai = 0
    fig, ax = plt.subplots()
    for ai in range(N_agents):
        # ax.plot(list_agents[ai].energy_trajectory, '.-')
        ax.plot(calories_cumulative_vec[ai,:], '.-')
    # ax.set_title('Agent ' + str(ai))
    ax.set_xlabel('Time step')
    ax.set_ylabel('Calories acquired')
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

#%% plot food reward map for one agent

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
fig, ax = plt.subplots()
sns.heatmap(ax=ax, data=f_otheragents_2d, center=0, cbar=True)
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
        value_heatmap = np.reshape(list_agents[ai].value_tlrajectory[:, ti], [edge_size,edge_size]) 
        sns.heatmap(ax=ax[r,c], data=value_heatmap,  center=0, vmin=-1, vmax=1, square=True)
        ax[r,c].set_title('value for agent ' + str(ai))


#%%
# ************************* CREATE USER INTERFACE ***************************

#scale the color bar based on the c weights (TO DO: find a better way to do this )
vmin = np.min(np.min(c_weights), 0)
vmax = np.max(np.max(c_weights), 0)
vmin = -1
vmax = 1

fig_ui, (ax_main, ax_value) = plt.subplots(1, 2, figsize=(10, 4))

ax_main.set_title('Environment')
# phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])
sns.heatmap(ax=ax_main, data=food_init_loc_2d, vmin=vmin, vmax=vmax, center=0, square=True, cbar=False)

# Choose one agent to highlight in magenta. Plot in an adjacent 
# panel the value function of this agent evolving over time 
z = 0 # index of highlighted agent
value_heatmap = np.reshape(list_agents[z].value_trajectory[:, 0], [edge_size,edge_size]) 
sns.heatmap(ax=ax_value, data=value_heatmap, vmin=vmin, vmax=vmax, center=0, square=True, cbar=True)
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
    sns.heatmap(ax=ax_main, data=food_heatmap, vmin=vmin, vmax=vmax, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
    
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
    sns.heatmap(ax=ax_value, data=value_heatmap, vmin=vmin, vmax=vmax, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
    
    return list_plot_agents #, plot_food

fig_ui.tight_layout()

ani = animation.FuncAnimation(fig_ui, update_plot, frames=N_timesteps,
                              fargs=(list_agents, food_trajectory, edge_size)) #, list_food_loc_id, edge_size))

#%%
saveMovie = False
from pathlib import Path
if saveMovie:
    filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_foodonly_v2.gif"
    filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_simple_explorebias10"
    
    my_file = Path(filepath + ".gif")
    # saving to m4 using ffmpeg writer
    if my_file.exists():
        filepath = filepath + '_v2'

    ani.save(filepath + ".gif", dpi=300, writer=animation.PillowWriter(fps=2))
    
    # ## saving as an mp4
    filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_mp4.mp4"
    # ani.save(filename, writer=animation.FFMpegWriter(fps=30) )
 