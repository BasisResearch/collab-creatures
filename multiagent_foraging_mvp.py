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
reload(util)
reload(figures)

figures.setup_fig()
plt.close('all')
plot_initial_world = False
plot_T = False

# ************** CREATE ENVIRONMENT ***************************

edge_size = 30 # grid world has dimensions edge_size x edge_size
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

N_food = 2
phi_food = np.zeros([N_states, 1]) 
list_food_loc_id = np.random.permutation(np.arange(N_states))[:N_food] # randomly choose K locations to place new food 
phi_food[list_food_loc_id] = 1 

food_2d_locs = np.reshape(phi_food, [edge_size,edge_size])

if plot_initial_world:
    fig_env, ax_env = plt.subplots()
    plt.imshow(food_2d_locs)

# food_loc_id = loc_id_arr[np.random.randint(edge_size**2)] # pick a random loc_id
# food_loc_id = edge_size**2 - 1 #put the food as far out as possible


# for i in range(N_food):
#     x, y = util.state_to_2Dloc(list_food_loc_id[i], edge_size)
#     ax_env.plot(x, y, 'ro')


# ----------------------- Add agents -----------------------------------
N_agents = 9
N_timesteps = 50
list_agents = []
arr_loc_id_allagents = np.zeros(N_agents)
phi_agents = np.zeros([N_states, 1]) # locations occupied by agents 

for i in range(N_agents):
    new_agent = TreeWorld.SimpleAgent(T_prob, N_states, N_timesteps=N_timesteps, discount_factor=0.8)
    list_agents.append(new_agent)
    
    current_loc_id = np.random.randint(N_states) # pick a random location]
    new_agent.state_trajectory[0] = current_loc_id
    
    # update which locations are occupied by agents 
    arr_loc_id_allagents[i] = current_loc_id   # array of state IDs
    phi_agents[current_loc_id] = 1                    # one-hot vector
    
    
# ************************** RUN SIMULATION  ***************************

for t in range(N_timesteps-1):
    ## ---------------------Update environment----------------------------
    # features = phi_food# (N_features, N_states) matrix, each row being 
    
    ## ---------------------Update agents ---------------------------------
    
    for i, agent in enumerate(list_agents):
        # sum_weighted_features = agent.c.T @ features 
        prev_state = int(agent.state_trajectory[t])
        c_food = 1
        c_neighbors = -1
        
        
        phi_agents[prev_state] = 0 # move out of previous location

        agent.phi_neighbors = phi_agents  # assume this agent knows locations of all other agents
        
        sum_weighted_features = c_food * phi_food + c_neighbors * agent.phi_neighbors
        
        value = agent.SR @ sum_weighted_features         # (N_states, 1) vector
        
        next_state = np.argmax(value * T_eligible[:, prev_state, np.newaxis]) # TO DO: change this to sampling probabilistically 
        
        agent.state_trajectory[t+1] = next_state        # scalar 
        agent.value_trajectory[:, t+1] = value.flatten()          # (N_states, 1) vector
        
        phi_agents[next_state] = 1                     # move into new location 
    

#%% plot social reward map for one agent
# fig, ax = plt.subplots()
# social_reward_map = agent.social_reward_given_location(arr_loc_id_allagents, edge_size)
# sns.heatmap(ax=ax, data=social_reward_map, center=0, cbar=True)
# ax.set_title('social reward map')

#%%  #------- plot value function for all agents at a specific time point -------

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

fig_ui, (ax_main, ax_value) = plt.subplots(1, 2, figsize=(16, 6))

ax_main.set_title('Environment')
# ax_main.imshow(food_2d_locs)
sns.heatmap(ax=ax_main, data=food_2d_locs, vmin=-1, vmax=1, center=0, square=True, cbar=False)

# TO DO: pick one agent to highlight in a different color. Plot in an adjacent 
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
    one_plot, = ax_main.plot([], [], marker)   
    list_plot_agents.append(one_plot)


def update_plot(frame, list_agents, edge_size): #, list_food_loc_id, edge_size):
    for  i, agent in enumerate(list_agents):
        state = int(agent.state_trajectory[frame])
        x, y = util.state_to_2Dloc(state, edge_size)
        list_plot_agents[i].set_data(x, y)
        
    # x_food, y_food = util.state_to_2Dloc(list_food_loc_id, edge_size)
    # plot_food.set_data(x_food, y_food)
    
    # value function of one agent 
    ax_value.cla()
    value_heatmap = np.reshape(list_agents[z].value_trajectory[:, frame], [edge_size,edge_size]) 
    sns.heatmap(ax=ax_value, data=value_heatmap, vmin=-1, vmax=1, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
    
    return list_plot_agents #, plot_food

ani = animation.FuncAnimation(fig_ui, update_plot, frames=N_timesteps,
                              fargs=(list_agents, edge_size)) #, list_food_loc_id, edge_size))

# saving to m4 using ffmpeg writer
filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_foodonly_v2.gif"
filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_food_and_otheragents_jump.gif"

ani.save(filename, dpi=300, writer=animation.PillowWriter(fps=2))

# ## saving as an mp4
filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_mp4.mp4"
# ani.save(filename, writer=animation.FFMpegWriter(fps=30) )
 