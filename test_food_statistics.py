# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:51:40 2023

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

edge_size = 30 # grid world has dimensions edge_size x edge_size
N_states = edge_size ** 2
x_arr, y_arr, loc_id_arr = util.create_2Dgrid(edge_size)

    
# ---------------------- Simulation parameters ------------------------------
N_sims = 1
N_timesteps = 2#1 total duration of the simulation in frames 

#food statistics and dynamics 
food_depletion_rate = 0.5
# food_statistics_type = "poisson"
food_statistics_type = "static"



if food_statistics_type == "poisson": 
    food_units_per_frame = 16
    N_total_food_units = np.ceil(food_units_per_frame * N_timesteps)# start with this being equal to the number of frames 
    # num_food_units_per_appearance = np.array([1, 4, 9, 16])
    patch_dim = 1
    num_food_units_per_patch = patch_dim ** 2
    N_food = num_food_units_per_patch
    rate_of_foodpatch_appearance = N_total_food_units / (num_food_units_per_patch * N_timesteps)# appearances per frame
    # rate_of_foodpatch_appearance = 1 / num_food_units_per_appearance
elif food_statistics_type == "static":
    N_food_units_total = 16
    patch_dim = 1
    N_units_per_patch = patch_dim ** 2
    N_patches = np.ceil(N_food_units_total / N_units_per_patch).astype(int)


# consider making a Simulation class and saving data from each simulation inside the object 
# or creating a pandas data frame 
for si in range(N_sims):
    

    phi_food = np.zeros([N_states, 1]) # indicator vector showing which locations are occupied by food. 
    food_calories_by_loc = np.zeros([N_states, 1]) # amount of food at each location in units of calories 
    food_trajectory = np.zeros([N_states, N_timesteps]) # track food calories over time 
    food_init_loc_2d = np.reshape(phi_food, [edge_size,edge_size])
    
    # genereate sequence of food appearance events 
    if food_statistics_type == "poisson":
        food_appearance_events = util.generate_poisson_events(rate_of_foodpatch_appearance, N_timesteps)
    
    elif food_statistics_type == "static": # drop food into environment only once, at the first time step 
        food_appearance_events = np.zeros(N_timesteps)
        food_appearance_events[0] = num_patches #

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
            food_calories_by_loc[list_newfood_loc_1d] = 20 

    
    # ************************** RUN SIMULATION  ***************************
    
    for ti in range(N_timesteps-1):
        ## ---------------------Update environment----------------------------
        # features = phi_food# (N_features, N_states) matrix, each row being 
        
        #Update food energy 
        # food occupied by an agent decays over time 

        
        if food_statistics_type == "poisson":
            delta_food_calories = food_depletion_rate * food_calories_by_loc    # change in magnitude of food amount at this time step 
            food_calories_by_loc -= delta_food_calories  # only subtract food calories in locations occupied by agents
            phi_food = food_calories_by_loc  > 0.01 # update indicator  vector for food locations 
            
            if food_appearance_events[ti]: # if food appears at this time step
                # choose a random (x, y) location to place the upper left corner of the patch. 
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
                

                
            
        # # randomly add a new food patch every several time steps  
        # if ti % epoch_dur == 0:
        #     list_newfood_loc_id = np.random.permutation(np.arange(N_states))[:N_food]
        #     phi_food[list_newfood_loc_id] = 1
        #     food_calories_by_loc[list_newfood_loc_id] = 20 # add a fixed number of calories to each new food location 
        #     # TO DO: make phi_food a calorie count (randomly pick between a range of calories) 
            
        # save food trajectory for plotting - how much food is in each location at each time step? 
        food_trajectory[:, ti+1] = food_calories_by_loc.flatten()    # (N_states, N_timesteps)   # save as a sparse matrix?

        # fig, ax = plt.subplots(figsize=(4, 4))
        # sns.heatmap(ax=ax, data=food_calories_by_loc, center=0, square=True, cbar=False)

        
        # ax_main.set_title('Environment')
        # phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])

#%% Plot initial food 
vmin, vmax = 0, 1

fig, ax = plt.subplots(figsize=(4, 4))

# ax_main.set_title('Environment')
# phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])
sns.heatmap(ax=ax, data=food_init_loc_2d, vmin=vmin, vmax=vmax, center=0, 
            yticklabels=False, xticklabels=False, square=True, cbar=False)

fig.tight_layout()

#%%
# ************************* CREATE USER INTERFACE ***************************

vmin = -1
vmax = 1

fig_ui, ax_main = plt.subplots(figsize=(4, 4))

# ax_main.set_title('Environment')
# phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])
sns.heatmap(ax=ax_main, data=food_init_loc_2d, vmin=vmin, vmax=vmax, center=0, square=True, cbar=False)


def update_plot(frame, food_trajectory, edge_size): #, list_food_loc_id, edge_size):
    fig_ui.suptitle('frame ' + str(frame) + ' / ' + str(N_timesteps))
    # food locations and quantities
    food_heatmap = np.reshape(food_trajectory[:, frame], [edge_size,edge_size]) 
    sns.heatmap(ax=ax_main, data=food_heatmap, vmin=vmin, vmax=vmax, center=0, square=True, cbar=False) #, cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})
    
    return 

fig_ui.tight_layout()

ani = animation.FuncAnimation(fig_ui, update_plot, frames=N_timesteps,
                              fargs=(food_trajectory, edge_size)) #, list_food_loc_id, edge_size))

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
 