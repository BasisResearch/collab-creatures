# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:16:45 2023

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

starttime = time.time()

# ************** CREATE ENVIRONMENT ***************************

edge_size = 20 # grid world has dimensions edge_size x edge_size
x_arr, y_arr, loc_id_arr = util.create_2Dgrid(edge_size)

# --------------- Build transition matrix-----------------------------

N_states = edge_size ** 2
T = np.zeros([N_states, N_states])

# compute eligible state transitions with a Euclidean distance rule 
# (up, down, left ,right)
for i in range(N_states):
  for j in range(N_states):
    T[i,j] = ( np.sqrt( (x_arr[j] - x_arr[i])**2 + (y_arr[j] - y_arr[i])**2 ) ) <= 1

T_eligible = T # save the binary representation 
T_prob = T / np.sum(T, axis=0, keepdims=True) # normalization so elements represent probabilities 
 
if plot_T: 
    plt.figure()
    plt.title('Eligible state transitions')
    plt.imshow(T_eligible)
    
    plt.figure()
    plt.title('State transition probabilities')
    plt.imshow(T)

# ---------------------- Add food rewards -----------------------------------
# TO DO: design a model of food dynamics 

# Data structures to consider: 
# 1) N_states x 1 one-hot vector indicating locations containing rewards
# 2) N_states x 1 vector with magnitude of reward at each state/locations
# 3) array or list of loc_ids (locations) containing rewards 
N_timesteps = 50

N_food = 2
f_food = np.zeros([N_states, 1]) 
food_trajectory = np.zeros([N_states, N_timesteps])
list_food_loc_id = np.random.permutation(np.arange(N_states))[:N_food] # randomly choose K locations to place new food 
f_food[list_food_loc_id] = 1 # put one food item in the respective locations
food_init_loc_2d = np.reshape(f_food, [edge_size,edge_size])

#food dynamics
food_depletion_rate = 0.5 
epoch_dur = 10 # add new food in random locations every epoch_dur time steps

N_predators = 4
phi_predator = np.zeros([N_states, 1]) 
list_predator_loc_id = np.random.permutation(np.arange(N_states))[:N_predators] # randomly choose K locations to place new food 
phi_predator[list_predator_loc_id] = 1
predator_2d_locs = np.reshape(phi_predator, [edge_size,edge_size])

if plot_initial_world:
    fig_env, ax_env = plt.subplots()
    plt.imshow(food_init_loc_2d)
    # plt.imshow(food_2d_locs + predator_2d_locs)

# food_loc_id = loc_id_arr[np.random.randint(edge_size**2)] # pick a random loc_id
# food_loc_id = edge_size**2 - 1 #put the food as far out as possible


# for i in range(N_food):
#     x, y = util.loc1Dto2D(list_food_loc_id[i], edge_size)
#     ax_env.plot(x, y, 'ro')


# ----------------------- Add agents -----------------------------------
N_agents = 4
list_agents = []
arr_loc_id_allagents = np.zeros(N_agents, dtype=int) # array containing location of each agent (index is agent ID)
phi_agents = np.zeros([N_states, 1]) # # one-hot vector indicating which locations are occupied by agents (index is loc ID)

for i in range(N_agents):
    new_agent = TreeWorld.SimpleAgent(T_prob, N_states, N_timesteps=N_timesteps, discount_factor=0.8)
    list_agents.append(new_agent)
    
    current_loc_id = np.random.randint(N_states) # pick a random location]
    new_agent.state_trajectory[0] = current_loc_id
    
    # update which locations are occupied by agents 
    arr_loc_id_allagents[i] = current_loc_id   # list
    phi_agents[current_loc_id] = 1                    # one-hot vector
    
    
# ************************** RUN SIMULATION  ***************************

for t in range(N_timesteps-1):
    ## ---------------------Update environment----------------------------
    # features = f_food# (N_features, N_states) matrix, each row being 
    
    #Update food feature vector 
    # food occupied by an agent decays over time 
    f_food = f_food - food_depletion_rate * f_food * phi_agents 
     
    # randomly add a new food patch every several time steps  
    if t % epoch_dur == 0:
        list_newfood_loc_id = np.random.permutation(np.arange(N_states))[:N_food]
        f_food[list_newfood_loc_id] = 1
        
    # save food trajectory for plotting
    food_trajectory[:, t+1] = f_food.flatten()    # (N_states, N_timesteps)   # save as a sparse matrix?
    
    ## ---------------------Update agents ---------------------------------
    
    for i, agent in enumerate(list_agents):
        # sum_weighted_features = agent.c.T @ features 
        prev_state = int(agent.state_trajectory[t])
        c_food = 2
        c_predator = 0
        c_otheragents = 0.5
        
        
        phi_agents[prev_state] = 0 # move out of previous location
        agent.phi_neighbors = phi_agents  # assume this agent knows the locations of all other agents
        
        # sum_weighted_features = c_food * f_food + c_otheragents * agent.phi_neighbors
        
        # compute weighted sum of features
        xloc_allagents, yloc_allagents = util.loc1Dto2D(arr_loc_id_allagents, edge_size)
        xloc_self, yloc_self = util.loc1Dto2D(prev_state, edge_size)
        f_otheragents_2d = agent.compute_reward_otheragents(xloc_allagents, yloc_allagents, xloc_self, yloc_self, edge_size)
        f_otheragents_1d = np.reshape(f_otheragents_2d, (N_states, 1))
        sum_weighted_features = c_food * f_food   + c_otheragents * f_otheragents_1d
        # sum_weighted_features = c_food * f_food  + c_predator * phi_predator #+ c_otheragents * f_otheragents_1d
        
        # compute value function
        value = agent.SR @ sum_weighted_features         # (N_states, 1) vector
        
        # select next action using the value and eligible states 
        # eligible states are those specified by the transition matrix and states not occupied by other agents
        eligible_states_id = np.nonzero(T_eligible[:, prev_state] * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
        value_eligible = value[eligible_states_id].flatten()   # value of eligible states plus some noise 
        # value_eligible += 0.01 * np.random.randn(value_eligible.shape[0]) # add some noise 
        
        # value_eligible = np.ones(value_eligible.shape[0]) + 0.01 * np.random.randn(value_eligible.shape[0])
        next_state = eligible_states_id[np.argmax(value_eligible)]  # DETERMINISTIC POLICY that works
        
        # #sample eligible states from a categorical distribution whose shape is based on the values 
        # # convert values into probabilities
        # prob_arr = value_eligible - np.min(value_eligible) # shift values so they are all positive and add some noise
        # prob_arr = prob_arr / np.sum(prob_arr) # normalize so they sum to 1
        # next_state = np.random.choice(eligible_states_id, p=prob_arr)
        
        agent.state_trajectory[t+1] = next_state        # scalar 
        agent.value_trajectory[:, t+1] = value.flatten()          # (N_states, N_timesteps)   
        
        phi_agents[next_state] = 1                     # move into new location 
        arr_loc_id_allagents[i] = next_state
    
endtime = time.time()
for i, agent in enumerate(list_agents):
    print(agent.state_trajectory)
print('simulation run time = ' + str(endtime - starttime))

#%% plot social reward map for one agent
fig, ax = plt.subplots()
sns.heatmap(ax=ax, data=f_otheragents_2d, center=0, cbar=True)
ax.set_title('social reward map')

#%%  #------- plot value function for all agents at a specific time point -------

if plotValueFuncAtTime: 
    t = 1 # which timestep to look at
    
    ncols = 3
    nrows = int(np.ceil(N_agents / ncols))
    grid_kws = {'wspace': 0.5}
    fig, ax = plt.subplots(nrows, ncols, gridspec_kw=grid_kws)
    plt.suptitle('Values at time t = ' + str(t))
    for  i, agent in enumerate(list_agents):
        r = int(np.ceil((i+1) / ncols) - 1)
        c = i % ncols
        value_heatmap = np.reshape(list_agents[i].value_trajectory[:, t], [edge_size,edge_size]) 
        sns.heatmap(ax=ax[r,c], data=value_heatmap,  center=0, vmin=-1, vmax=1, square=True)
        ax[r,c].set_title('value for agent ' + str(i))


#%%
# ************************* CREATE USER INTERFACE ***************************

fig_ui, (ax_main, ax_value) = plt.subplots(1, 2, figsize=(10, 4))

ax_main.set_title('Environment')
# f_food_2d = np.reshape(f_food, [edge_size,edge_size])
sns.heatmap(ax=ax_main, data=food_init_loc_2d, vmin=-1, vmax=1, center=0, square=True, cbar=False)

# Choose one agent to highlight in magenta. Plot in an adjacent 
# panel the value function of this agent evolving over time 
z = 0 # index of highlighted agent
value_heatmap = np.reshape(list_agents[z].value_trajectory[:, 0], [edge_size,edge_size]) 
sns.heatmap(ax=ax_value, data=value_heatmap, vmin=-1, vmax=1, center=0, square=True, cbar=True)
ax_value.set_title('Value function for one agent')

list_plot_agents = []
markers = ['co'] # color for all other agents 

# plot each agent's data
for  i, agent in enumerate(list_agents):
    if i == z:
        marker = 'mo'
    else:
        marker = markers[0]
    one_plot, = ax_main.plot([], [], marker, markersize=5)   
    list_plot_agents.append(one_plot)


def update_plot(frame, list_agents, food_trajectory, edge_size): #, list_food_loc_id, edge_size):
    fig_ui.suptitle('frame ' + str(frame) + ' / ' + str(N_timesteps))
    # food locations and quantities
    food_heatmap = np.reshape(food_trajectory[:, frame], [edge_size,edge_size]) 
    sns.heatmap(ax=ax_main, data=food_heatmap, vmin=-1, vmax=1, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
    
    # x_food, y_food = util.loc1Dto2D(list_food_loc_id, edge_size)
    # plot_food.set_data(x_food, y_food)
    
    # state of each agent 
    for  i, agent in enumerate(list_agents):
        state = int(agent.state_trajectory[frame])
        x, y = util.loc1Dto2D(state, edge_size)
        list_plot_agents[i].set_data(x, y)
    
    # value function of one agent 
    ax_value.cla()
    value_heatmap = np.reshape(list_agents[z].value_trajectory[:, frame], [edge_size,edge_size]) 
    sns.heatmap(ax=ax_value, data=value_heatmap, vmin=-1, vmax=1, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
    
    return list_plot_agents #, plot_food

fig_ui.tight_layout()

ani = animation.FuncAnimation(fig_ui, update_plot, frames=N_timesteps,
                              fargs=(list_agents, food_trajectory, edge_size)) #, list_food_loc_id, edge_size))

#%%
saveMovie = False
if saveMovie:
    # saving to m4 using ffmpeg writer
    filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_foodonly_v2.gif"
    filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_lazy_avoid_others.gif"
    
    ani.save(filename, dpi=300, writer=animation.PillowWriter(fps=2))
    
    # ## saving as an mp4
    filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_mp4.mp4"
    # ani.save(filename, writer=animation.FFMpegWriter(fps=30) )
 