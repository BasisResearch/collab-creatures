# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:16:47 2023

Modified version of cvpr_submission_sims.py. Runs a single simulation

"""

import TreeWorld
import seaborn as sns
import numpy as np
import pandas as pd
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
<<<<<<< HEAD
saveData = True
directory_data = "simulated_data/"
filename_data = "new_sim_2023-08-14_test"
=======
saveData = False
directory_data = "simulated_data/"
filename_data = "new_sim"
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)

doAnimation = True
saveMovie = False
directory_movie = "movies/"
filename_movie = "new_movie"

# Turn on/off plots an animations
plot_initial_world = False
plot_T = False
plotValueFuncAtTime = False
do_plot_dist_to_neighbors = False
do_plot_numfailed = False
do_plot_timetofood = False
do_plot_calories = False
do_plot_visible_food = False
do_plot_value_otheragents = False
do_plot_model_internals = False
doPrintAgentStateTrajectories = False

<<<<<<< HEAD
# RU: needed to comment out these two to be able to run without errors!
# figures.setup_fig()
# plt.close("all")
=======

figures.setup_fig()
plt.close("all")
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)

# ---------------------- Simulation parameters ------------------------------

## ru: so you run them multiple times, but every time with the same parameters?
## ru: for the paper it would be great to randomize them a bit

<<<<<<< HEAD
N_sims = 3
N_timesteps = 150
N_agents = 6
=======
N_timesteps = 3
N_agents = 2
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)

# Food and environment parameters
food_statistics_types = [
    "drop_food_once",
<<<<<<< HEAD
    "respawn",
    "regular_intervals",
    "poisson",
]
food_statistics_type = "respawn"
# food_statistics_type = "regular_intervals"
=======
    "replenish_only_after_depleted",
    "regular_intervals",
    "poisson",
]
food_statistics_type = "drop_food_once"
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
N_food_units_total = 16
patch_dim = 4  # a patch has dimensions (patch_dim x patch_dim )
N_units_per_patch = patch_dim**2
N_patches = np.ceil(N_food_units_total / N_units_per_patch).astype(int)
<<<<<<< HEAD
calories_acquired_per_unit_time = (
    5  # when an agent is at a food location, it gains this many calories per time step
)
epoch_dur = N_timesteps  # add new food in random locations every epoch_dur time steps
=======
calories_acquired_per_unit_time = 5  # when an agent is at a food location, it gains this many calories per time step
epoch_dur = (
    N_timesteps  # add new food in random locations every epoch_dur time steps
)
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)

# Agent parameters
# doShareFoodInfo = True # Binary variable - are the birds communicating or not?
max_step_size = 3
<<<<<<< HEAD
sight_radius = 3
=======
sight_radius = 5
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
energy_init = 50
discount_factor = 0.9
caloric_cost_per_unit_dist = 1
doProbabilisticPolicy = True
doSoftmaxPolicy = True
<<<<<<< HEAD
exploration_bias = 0.0001
# agent_type = 'ignorers'
agent_type = "communicators"
# agent_type = 'followers'

if agent_type == "hungry":
    c_food_self = 1
    c_food_others = 0 # to what extent does the bird use information from other birds?
    c_otheragents = 0
    c_group = 0
elif agent_type == "communicators":
    # Note the implementation of communicators here is slightly different from
    # that of the CVPR workshop paper. For the CVPR paper's implementation
    # see cvpr_submission_sims.py
    c_food_self = 0.5
    c_food_others = 0.5
    c_otheragents = 0
    c_group = 0
elif agent_type == "high_trust":
    c_food_self = 0
    c_food_others = 1
=======
exploration_bias = 0.005
# agent_type = 'ignorers'
agent_type = 'communicators'
# agent_type = 'followers'

if agent_type == 'ignorers':
    c_food_self = 0.9
    c_food_others = 0.1  # to what extent do the birds care about information from other birds?
    c_otheragents = 0
    c_group = 0
elif agent_type == 'communicators':
    # Note the implementation of communicators here is slightly different from
    # that of the CVPR workshop paper. For the CVPR paper's implementation 
    # see cvpr_submission_sims.py
    c_food_self = 0.5
    c_food_others = 0.5   
    c_otheragents = 0
    c_group = 0
elif agent_type == 'followers':
    c_food_self = 0.1
    c_food_others = 0.9   
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
    c_otheragents = 0
    c_group = 0

c_weights = [c_food_self, c_food_others, c_otheragents, c_group]

# Quantities to track
<<<<<<< HEAD
allsimDFs = [] # list of each simulation's set of data frames. 
x_agents_all = np.zeros([N_agents, N_timesteps])
y_agents_all = np.zeros([N_agents, N_timesteps])
=======
x_agents_all = np.zeros([N_agents, N_timesteps])
y_agents_all = np.zeros([N_agents, N_timesteps]) 
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
agent_locs_1d_all = np.zeros([N_agents, N_timesteps])
dist_to_nearest_neighbor_all = np.zeros([N_agents, N_timesteps])
calories_acquired_all = np.zeros([N_agents, N_timesteps])
time_to_first_food_all = np.zeros([N_agents, 1])

# ************** CREATE ENVIRONMENT ***************************
edge_size = 30  # grid world has dimensions edge_size x edge_size
x_arr, y_arr, loc_1d_arr = util.create_2Dgrid(edge_size)

# --------------- Build transition matrix-----------------------------

N_states = edge_size**2
T = np.zeros([N_states, N_states])

# compute eligible state transitions with a Euclidean distance rule
# (up, down, left ,right)
for i in range(N_states):
    for j in range(N_states):
        T[i, j] = (
            np.sqrt((x_arr[j] - x_arr[i]) ** 2 + (y_arr[j] - y_arr[i]) ** 2)
        ) <= max_step_size  # make this bigger to include more eligible states!!!

T_eligible = T  # save the binary representation
T_prob = T / np.sum(
    T, axis=0, keepdims=True
)  # normalization so elements represent probabilities

if plot_T:
    plt.figure()
    plt.title("Eligible state transitions")
    plt.imshow(T_eligible)

    plt.figure()
    plt.title("State transition probabilities")
    plt.imshow(T_prob)


# ----------------------------------------------------------------------------
# consider making a Simulation class and saving data from each simulation inside the object
# or creating a pandas data frame
starttime = time.time()

<<<<<<< HEAD
for si in range(N_sims):
    
    # ---------------------- Add food rewards -----------------------------------
    # TO DO: design a model of food dynamics
    
    # Data structures to consider:
    # 1) N_states x 1 one-hot vector indicating locations containing rewards
    # 2) N_states x 1 vector with magnitude of reward at each state/locations
    # 3) array or list of loc_ids (locations) containing rewards
    
    phi_food = np.zeros(
        [N_states, 1]
    )  # indicator vector showing which locations are occupied by food.
    food_calories_by_loc = np.zeros(
        [N_states, 1]
    )  # amount of food at each location in units of calories
    food_trajectory = np.zeros([N_states, N_timesteps])  # track food calories over time
    # list_food_loc_id = np.zeros([N_food]) # array of locations where there is food
    # list_food_loc_id = np.random.permutation(np.arange(N_states))[:N_food] # randomly choose K locations to place new food
    # phi_food[list_food_loc_id] = 1 # put one food item in the respective locations
    food_init_loc_2d = np.reshape(phi_food, [edge_size, edge_size])
    
    # if food_statistics_type == "drop_food_once":
    # Add inital food patches to the environment
=======
# ---------------------- Add food rewards -----------------------------------
# TO DO: design a model of food dynamics

# Data structures to consider:
# 1) N_states x 1 one-hot vector indicating locations containing rewards
# 2) N_states x 1 vector with magnitude of reward at each state/locations
# 3) array or list of loc_ids (locations) containing rewards

phi_food = np.zeros(
    [N_states, 1]
)  # indicator vector showing which locations are occupied by food.
food_calories_by_loc = np.zeros(
    [N_states, 1]
)  # amount of food at each location in units of calories
food_trajectory = np.zeros(
    [N_states, N_timesteps]
)  # track food calories over time
# list_food_loc_id = np.zeros([N_food]) # array of locations where there is food
# list_food_loc_id = np.random.permutation(np.arange(N_states))[:N_food] # randomly choose K locations to place new food
# phi_food[list_food_loc_id] = 1 # put one food item in the respective locations
food_init_loc_2d = np.reshape(phi_food, [edge_size, edge_size])

if food_statistics_type == "drop_food_once":
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
    for pi in range(N_patches):
        x_start = np.random.randint(0, edge_size - patch_dim)
        y_start = np.random.randint(0, edge_size - patch_dim)
        # generate (x,y) coordinates for each food unit in the patch
<<<<<<< HEAD
        x_range, y_range = np.arange(x_start, x_start + patch_dim), np.arange(
            y_start, y_start + patch_dim
        )
=======
        x_range, y_range = np.arange(
            x_start, x_start + patch_dim
        ), np.arange(y_start, y_start + patch_dim)
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
        x_locs, y_locs = np.meshgrid(x_range, y_range, indexing="xy")
        # convert to 1D locations
        list_newfood_loc_1d = util.loc2Dto1D(
            x_locs.flatten(), y_locs.flatten(), edge_size
        )
<<<<<<< HEAD
    
=======

>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
        # update food tracking variables
        phi_food[list_newfood_loc_1d] = 1
        food_calories_by_loc[
            list_newfood_loc_1d
        ] = 20  # add a fixed number of calories to each new food location
<<<<<<< HEAD
    
    # if food_statistics_type == 'sequential': # new food appears only when old food is completely depleted
    
    N_predators = 0
    w_predators = np.zeros([N_states, 1])  # vector indicating location of predators
    list_predator_loc_id = np.random.permutation(np.arange(N_states))[
        :N_predators
    ]  # randomly choose K locations to place new food
    w_predators[list_predator_loc_id] = 1
    predator_2d_locs = np.reshape(w_predators, [edge_size, edge_size])
    
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
    loc_1d_allagents = np.zeros(
        N_agents, dtype=int
    )  # array containing location of each agent (index is agent ID)
    phi_agents = np.zeros(
        [N_states, 1]
    )  # # one-hot vector indicating how many agents are in each location (index is loc ID)
    
    # matrix tracking energy acquisition over time, used for determining fitness of the species
    calories_acquired_mat = np.zeros([N_agents, N_timesteps])
    calories_expended_mat = np.zeros([N_agents, N_timesteps])
    calories_total_mat = np.zeros([N_agents, N_timesteps])
    calories_cumulative_vec = np.zeros([N_agents, N_timesteps])
    
    # initialize the agents
    for ai in range(N_agents):
        new_agent = TreeWorld.SimpleAgent(
            T_prob,
            N_states,
            N_timesteps=N_timesteps,
            discount_factor=discount_factor,
            energy_init=energy_init,
            sight_radius=sight_radius,
        )
        list_agents.append(new_agent)
    
        current_loc_id = np.random.randint(N_states)  # pick a random location]
        new_agent.state_trajectory[0] = current_loc_id
    
        # update which locations are occupied by agents
        loc_1d_allagents[ai] = current_loc_id  # list
        phi_agents[current_loc_id] += 1  # add an agent to this location
    
        # agent.energy_trajectory[0] = 50 # each agent starts with 50 calories --> done inside the class
    
    calories_total_mat[:, 0] = energy_init  # each agent starts with 50 calories
    
    # ************************** RUN SIMULATION  ***************************
    
    for ti in range(N_timesteps - 1):
        print(" time step " + str(ti))
    
        ## ---------------------Update environment----------------------------
        # features = phi_food# (N_features, N_states) matrix, each row being
    
        # Update agent calorie levels
        # food occupied by an agent decays over time
        # delta_food_calories_total = food_depletion_rate * food_calories_by_loc * phi_agents # only subtract food calories in locations occupied by agents, scaled by the number of agents
        delta_food_calories_total = calories_acquired_per_unit_time * phi_agents
        # rectify the calorie count for the food locations that will hit negative calories
        is_overdepleted = (
            delta_food_calories_total > food_calories_by_loc
        )  # find locations where the calorie count will hit negative values (we'll set the calorie count to 0)
        delta_food_calories_total[is_overdepleted] = food_calories_by_loc[is_overdepleted]
    
        food_calories_by_loc -= delta_food_calories_total
        phi_food = (
            food_calories_by_loc > 0.01
        )  # update indicator  vector for food locations
    
        if food_statistics_type == "respawn":
            thresh = 2 # if the number of food units falls below this number, respawn by adding N_food_units_total new food units
            if np.sum(phi_food) < 2:
                for pi in range(N_patches):
                    x_start = np.random.randint(0, edge_size - patch_dim)
                    y_start = np.random.randint(0, edge_size - patch_dim)
                    # generate (x,y) coordinates for each food unit in the patch
                    x_range, y_range = np.arange(x_start, x_start + patch_dim), np.arange(
                        y_start, y_start + patch_dim
                    )
                    x_locs, y_locs = np.meshgrid(x_range, y_range, indexing="xy")
                    # convert to 1D locations
                    list_newfood_loc_1d = util.loc2Dto1D(
                        x_locs.flatten(), y_locs.flatten(), edge_size
                    )
    
                    # update food tracking variables
                    phi_food[list_newfood_loc_1d] = 1
                    food_calories_by_loc[
                        list_newfood_loc_1d
                    ] = 20  # add a fixed number of calories to each new food location
    
        if food_statistics_type == "regular_intervals":
            # TO DO: Fix this part. NOT WORKING YET 
            # randomly add a new food patch every interval_food time steps
            interval_food = 10 
            if ti % epoch_dur == interval_food:
                list_newfood_loc_id = np.random.permutation(np.arange(N_states))[:N_patches]
                phi_food[list_newfood_loc_id] = 1
                food_calories_by_loc[
                    list_newfood_loc_id
                ] = 20  # add a fixed number of calories to each new food location
                # TO DO: make phi_food a calorie count (randomly pick between a range of calories)
    
        # save food trajectory for plotting - how much food is in each location at each time step?
        food_trajectory[
            :, ti + 1
        ] = (
            food_calories_by_loc.flatten()
        )  # (N_states, N_timesteps)   # save as a sparse matrix?
    
        # update predator locations
    
        ## ---------------------Update agents ---------------------------------
    
        for ai, agent in enumerate(list_agents):
            # sum_weighted_features = agent.c.T @ features
            prev_loc_1d = int(agent.state_trajectory[ti])  # agent's current location
    
            # ------ update energy consequences of previous time step's actions --------
    
            # update agent's total energy based on amount of food at previous location
            # transfer calories from food to agent
            calories_acquired_mat[ai, ti] = (
                delta_food_calories_total[prev_loc_1d] / phi_agents[prev_loc_1d][0]
            )[0]
            # RU: added [0] to ensure its a scalar; otherwise deprecation warning
            # RU: make sure that this is in line with your intentions
    
            #   # if there were N agents at that location, it gets 1/N portion of the calories
            agent.energy_total += calories_acquired_mat[ai, ti]
            calories_cumulative_vec[ai, ti + 1] = (
                calories_cumulative_vec[ai, ti] + calories_acquired_mat[ai, ti]
            )  # only tracks calories acquired?
    
            # # remove this agent from the list of surviving agents if it's energy reaches zero
            # if agent.energy_total <= 0:
            #     list_deceased_agents = list_surviving_agents.pop(ai)  # be careful b/c the rest of this for loop assumes all the agents are alive
    
            # -------------- Compute expected rewards, values, and make a decision --------------------------------
    
            phi_agents[prev_loc_1d] -= 1  # move out of previous location
    
            # EXPECTED REWARD RELATED TO OTHER AGENTS
            xloc_allagents, yloc_allagents = util.loc1Dto2D(loc_1d_allagents, edge_size)
            xloc_self, yloc_self = util.loc1Dto2D(prev_loc_1d, edge_size)
            # only include locations of agents outside of current location
            xloc_neighbors, yloc_neighbors = util.loc1Dto2D(
                loc_1d_allagents[loc_1d_allagents != prev_loc_1d], edge_size
            )
    
            # expected reward at each location based on proximity to other agents
            w_otheragents_2d = agent.reward_function_otheragents(
                xloc_neighbors, yloc_neighbors, xloc_self, yloc_self, edge_size
            )
            w_otheragents_1d = np.reshape(w_otheragents_2d, (N_states, 1))
    
            # EXPECTED REWARD RELATED TO CENTER OF MASS
            xloc_otheragents = np.delete(
                xloc_allagents, ai
            )  # remove this agent's own location from the list
            yloc_otheragents = np.delete(yloc_allagents, ai)  #
            if N_agents > 1:
                xloc_centerofmass, yloc_centerofmass = util.center_of_mass(
                    xloc_otheragents, yloc_otheragents
                )
            else:
                xloc_centerofmass, yloc_centerofmass = xloc_self, yloc_self
    
            # expected reward of each location based on this agent's distance from center of mass of the group
            w_groupcenterofmass = np.zeros([edge_size, edge_size])
            w_groupcenterofmass[int(yloc_centerofmass), int(xloc_centerofmass)] = 0.5
            w_groupcenterofmass = np.reshape(w_groupcenterofmass, (N_states, 1))
    
            # VISIBILITY CONSTRAINTS
            phi_visible_mat = agent.compute_visible_locations(
                xloc_self, yloc_self, edge_size
            )
            phi_visible = np.reshape(phi_visible_mat, (N_states, 1))
    
            # RU: let me make sure I understand:
            # The info from other agents is just info about whether
            # there is food at their exact locations,
            # not whether there is food within the other birds
            # visibility range, right?
    
            # get information from other agents about whether there is food at their locations
            # if doShareFoodInfo:
            #     phi_visible[loc_1d_allagents] = 1   # can this agent see the locations of other agents?
            # It's not quite communication between agents yet because there is no
            # capacity for misrepresentation here - the agent simply has information about other agent's locations.
            # The info from other agents should be represented separately from the agent's own information.
            # Then you can change communication parameters such as fidelity of info transmitted (add noise)
    
            # EXPECTED REWARD RELATED TO FOOD
            w_food = phi_food * phi_visible  # expected food reward at each location
            w_food_others = (
                phi_food * phi_agents
            )  # making food info from other agents a separate feature with separate weights
    
            sum_weighted_features = (
                c_weights[0] * w_food
                + c_weights[1] * w_food_others
                + c_weights[2] * w_otheragents_1d
                + c_weights[3] * w_groupcenterofmass
            )
    
            # sum_weighted_features = c_food_self * phi_food  + c_predator * phi_predator #+ c_otheragents * w_otheragents_1d
    
            # VALUE FUNCTION
            value = agent.SR @ sum_weighted_features  # (N_states, 1) vector
    
            # POLICY: select next action using the value and eligible states
            # eligible states are those specified by the transition matrix. Can constrain further to exclude states not occupied by other agents
            eligible_states_id = np.nonzero(T_eligible[:, prev_loc_1d])[
                0
            ]  # * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
            value_eligible = value[
                eligible_states_id
            ].flatten()  # value of eligible states plus some noise
    
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
                    prob_arr += (
                        np.mean(value_eligible)
                        * 0.001
                        * np.random.rand(value_eligible.shape[0])
                    )
                    prob_arr = prob_arr / np.sum(prob_arr)  # normalize so they sum to 1
    
                next_loc_1d = np.random.choice(eligible_states_id, p=prob_arr)
    
            else:
                next_loc_1d = eligible_states_id[
                    np.argmax(value_eligible)
                ]  # DETERMINISTIC POLICY that works
    
            # ------ Locations of agents and rewards at each time point --------
            xloc_prev, yloc_prev = util.loc1Dto2D(prev_loc_1d, edge_size)
            xloc_next, yloc_next = util.loc1Dto2D(next_loc_1d, edge_size)
    
            x_agents_all[ai, ti] = xloc_prev
            y_agents_all[ai, ti] = yloc_prev
    
            # ------- compute energy cost of moving to new location --------------
            dist_traveled = np.sqrt(
                (xloc_next - xloc_prev) ** 2 + (yloc_next - yloc_prev) ** 2
            )
            calories_expended_mat[ai, ti] = caloric_cost_per_unit_dist * dist_traveled
            agent.energy_total -= calories_expended_mat[ai, ti]
    
            # ------------- compute metrics for data analysis -----------------
            if len(list_agents) > 1:
                dist_to_neighbors = np.sqrt(
                    (xloc_otheragents - xloc_self) ** 2
                    + (yloc_otheragents - yloc_self) ** 2
                )
                dist_to_nearest_neighbor_all[ai, ti] = np.min(dist_to_neighbors)
    
            calories_acquired_all[ai, ti] = calories_acquired_mat[ai, ti]
    
            if phi_food[next_loc_1d][0]:
                agent.times_at_food.append(
                    ti + 1
                )  # add this frame to the list of frames where agent is at a food location
    
            # -------------------------------------------------------------------
    
            agent.state_trajectory[ti + 1] = next_loc_1d  # scalar
            agent.value_trajectory[:, ti + 1] = value.flatten()  # (N_states, N_timesteps)
            agent.energy_trajectory[ti + 1] = agent.energy_total
            calories_total_mat[ai, ti + 1] = (
                calories_total_mat[ai, ti]
                + calories_acquired_mat[ai, ti]
                - calories_expended_mat[ai, ti]
            )
    
            phi_agents[next_loc_1d] += 1  # move into new location
            loc_1d_allagents[ai] = next_loc_1d
    
    for ai, agent in enumerate(list_agents):
        # print(agent.times_at_food)
        if len(agent.times_at_food) == 0:
            time_to_first_food_all[
                ai
            ] = N_timesteps  # TO DO: figure out how to track agents who never reach food
        else:
            time_to_first_food_all[ai] = agent.times_at_food[
                0
            ]  # fetch the first time step where agent is at a food location
    
    endtime = time.time()
    
    if doPrintAgentStateTrajectories:
        for ai, agent in enumerate(list_agents):
            print(agent.state_trajectory)
    
    print("simulation run time = " + str(endtime - starttime))
    
    # -------Save locations of each agent and reward in data frames-----------
    # all_birdsDF:
    # all_rewardsDF:
    
    # Bird locations
    birds_all = []
    for ai in range(N_agents):
        single_agent = pd.DataFrame(
            {
                "x": x_agents_all[ai, :],
                "y": y_agents_all[ai, :],
                "time": range(1, N_timesteps + 1),
                "bird": ai + 1,
                "type": "random",
            }
        )
    
        birds_all.append(single_agent)
    
    all_birdsDF = pd.concat(birds_all)
    
    all_birdsDF.head()
    
    # Reward locations
    rewards_all = []
    for ti in range(N_timesteps):
        loc1D = [idx for idx, val in enumerate(food_trajectory[:, ti]) if val > 0]
        if len(loc1D) > 0:
            x_reward, y_reward = util.loc1Dto2D(loc1D, edge_size)
            single_time = pd.DataFrame(
                {
                    "x": x_reward,
                    "y": y_reward,
                    "time": np.repeat(ti, len(x_reward)),
                }
            )
            rewards_all.append(single_time)
    
    all_rewardsDF = pd.concat(rewards_all)
    
    print(all_birdsDF)
    print(all_rewardsDF)
    
    communicatorsDFs = [all_birdsDF, all_rewardsDF]
    allsimDFs.append(communicatorsDFs)
    
filename = agent_type + "SightRad" + str(sight_radius) + "PatchDim" + str(patch_dim) 
with open(filename + ".pkl", "wb") as f:
    pickle.dump(allsimDFs, f)

=======

# if food_statistics_type == 'sequential': # new food appears only when old food is completely depleted

N_predators = 0
w_predators = np.zeros(
    [N_states, 1]
)  # vector indicating location of predators
list_predator_loc_id = np.random.permutation(np.arange(N_states))[
    :N_predators
]  # randomly choose K locations to place new food
w_predators[list_predator_loc_id] = 1
predator_2d_locs = np.reshape(w_predators, [edge_size, edge_size])

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
loc_1d_allagents = np.zeros(
    N_agents, dtype=int
)  # array containing location of each agent (index is agent ID)
phi_agents = np.zeros(
    [N_states, 1]
)  # # one-hot vector indicating how many agents are in each location (index is loc ID)

# matrix tracking energy acquisition over time, used for determining fitness of the species
calories_acquired_mat = np.zeros([N_agents, N_timesteps])
calories_expended_mat = np.zeros([N_agents, N_timesteps])
calories_total_mat = np.zeros([N_agents, N_timesteps])
calories_cumulative_vec = np.zeros([N_agents, N_timesteps])

# initialize the agents
for ai in range(N_agents):
    new_agent = TreeWorld.SimpleAgent(
        T_prob,
        N_states,
        N_timesteps=N_timesteps,
        discount_factor=discount_factor,
        energy_init=energy_init,
        sight_radius=sight_radius,
    )
    list_agents.append(new_agent)

    current_loc_id = np.random.randint(N_states)  # pick a random location]
    new_agent.state_trajectory[0] = current_loc_id

    # update which locations are occupied by agents
    loc_1d_allagents[ai] = current_loc_id  # list
    phi_agents[current_loc_id] += 1  # add an agent to this location

    # agent.energy_trajectory[0] = 50 # each agent starts with 50 calories --> done inside the class

calories_total_mat[
    :, 0
] = energy_init  # each agent starts with 50 calories

# ************************** RUN SIMULATION  ***************************

for ti in range(N_timesteps - 1):
    print(" time step " + str(ti))

    ## ---------------------Update environment----------------------------
    # features = phi_food# (N_features, N_states) matrix, each row being

    # Update agent calorie levels
    # food occupied by an agent decays over time
    # delta_food_calories_total = food_depletion_rate * food_calories_by_loc * phi_agents # only subtract food calories in locations occupied by agents, scaled by the number of agents
    delta_food_calories_total = (
        calories_acquired_per_unit_time * phi_agents
    )
    # rectify the calorie count for the food locations that will hit negative calories
    is_overdepleted = (
        delta_food_calories_total > food_calories_by_loc
    )  # find locations where the calorie count will hit negative values (we'll set the calorie count to 0)
    delta_food_calories_total[is_overdepleted] = food_calories_by_loc[
        is_overdepleted
    ]

    food_calories_by_loc -= delta_food_calories_total
    phi_food = (
        food_calories_by_loc > 0.01
    )  # update indicator  vector for food locations

    # if food_statistics_type == "replenish_after_depletion":
    #     if np.sum(food_calories_by_loc) <= 0:

    if food_statistics_type == "regular_intervals":
        # randomly add a new food patch every several time steps
        if ti % epoch_dur == 0:
            list_newfood_loc_id = np.random.permutation(
                np.arange(N_states)
            )[:N_patches]
            phi_food[list_newfood_loc_id] = 1
            food_calories_by_loc[
                list_newfood_loc_id
            ] = 20  # add a fixed number of calories to each new food location
            # TO DO: make phi_food a calorie count (randomly pick between a range of calories)

    # save food trajectory for plotting - how much food is in each location at each time step?
    food_trajectory[
        :, ti + 1
    ] = (
        food_calories_by_loc.flatten()
    )  # (N_states, N_timesteps)   # save as a sparse matrix?

    # update predator locations

    ## ---------------------Update agents ---------------------------------

    for ai, agent in enumerate(list_agents):
        # sum_weighted_features = agent.c.T @ features
        prev_loc_1d = int(
            agent.state_trajectory[ti]
        )  # agent's current location

        # ------ update energy consequences of previous time step's actions --------

        # update agent's total energy based on amount of food at previous location
        # transfer calories from food to agent
        calories_acquired_mat[ai, ti] = (
            delta_food_calories_total[prev_loc_1d]
            / phi_agents[prev_loc_1d][0]
        )[0]
        # RU: added [0] to ensure its a scalar; otherwise deprecation warning
        # RU: make sure that this is in line with your intentions
        
        #   # if there were N agents at that location, it gets 1/N portion of the calories
        agent.energy_total += calories_acquired_mat[ai, ti]
        calories_cumulative_vec[ai, ti + 1] = (
            calories_cumulative_vec[ai, ti] + calories_acquired_mat[ai, ti]
        )  # only tracks calories acquired?

        # # remove this agent from the list of surviving agents if it's energy reaches zero
        # if agent.energy_total <= 0:
        #     list_deceased_agents = list_surviving_agents.pop(ai)  # be careful b/c the rest of this for loop assumes all the agents are alive

        # -------------- Compute expected rewards, values, and make a decision --------------------------------

        phi_agents[prev_loc_1d] -= 1  # move out of previous location

        # EXPECTED REWARD RELATED TO OTHER AGENTS
        xloc_allagents, yloc_allagents = util.loc1Dto2D(
            loc_1d_allagents, edge_size
        )
        xloc_self, yloc_self = util.loc1Dto2D(prev_loc_1d, edge_size)
        # only include locations of agents outside of current location
        xloc_neighbors, yloc_neighbors = util.loc1Dto2D(
            loc_1d_allagents[loc_1d_allagents != prev_loc_1d], edge_size
        )

        # expected reward at each location based on proximity to other agents
        w_otheragents_2d = agent.reward_function_otheragents(
            xloc_neighbors, yloc_neighbors, xloc_self, yloc_self, edge_size
        )
        w_otheragents_1d = np.reshape(w_otheragents_2d, (N_states, 1))

        # EXPECTED REWARD RELATED TO CENTER OF MASS
        xloc_otheragents = np.delete(
            xloc_allagents, ai
        )  # remove this agent's own location from the list
        yloc_otheragents = np.delete(yloc_allagents, ai)  #
        if N_agents > 1:
            xloc_centerofmass, yloc_centerofmass = util.center_of_mass(
                xloc_otheragents, yloc_otheragents
            )
        else:
            xloc_centerofmass, yloc_centerofmass = xloc_self, yloc_self

        # expected reward of each location based on this agent's distance from center of mass of the group
        w_groupcenterofmass = np.zeros([edge_size, edge_size])
        w_groupcenterofmass[
            int(yloc_centerofmass), int(xloc_centerofmass)
        ] = 0.5
        w_groupcenterofmass = np.reshape(
            w_groupcenterofmass, (N_states, 1)
        )
        
        # VISIBILITY CONSTRAINTS
        phi_visible_mat = agent.compute_visible_locations(
            xloc_self, yloc_self, edge_size
        )
        phi_visible = np.reshape(phi_visible_mat, (N_states, 1))


        #RU: let me make sure I understand: 
        #The info from other agents is just info about whether
        #there is food at their exact locations, 
        #not whether there is food within the other birds
        #visibility range, right?

        # get information from other agents about whether there is food at their locations
        # if doShareFoodInfo:
        #     phi_visible[loc_1d_allagents] = 1   # can this agent see the locations of other agents?
        # It's not quite communication between agents yet because there is no
        # capacity for misrepresentation here - the agent simply has information about other agent's locations.
        # The info from other agents should be represented separately from the agent's own information.
        # Then you can change communication parameters such as fidelity of info transmitted (add noise)

        # EXPECTED REWARD RELATED TO FOOD
        w_food = (
            phi_food * phi_visible
        )  # expected food reward at each location
        w_food_others = (
            phi_food * phi_agents
        )  # making food info from other agents a separate feature with separate weights

        sum_weighted_features = (
            c_weights[0] * w_food
            + c_weights[1] * w_food_others
            + c_weights[2] * w_otheragents_1d
            + c_weights[3] * w_groupcenterofmass
        )

        # sum_weighted_features = c_food_self * phi_food  + c_predator * phi_predator #+ c_otheragents * w_otheragents_1d

        # VALUE FUNCTION
        value = agent.SR @ sum_weighted_features  # (N_states, 1) vector

        # POLICY: select next action using the value and eligible states
        # eligible states are those specified by the transition matrix. Can constrain further to exclude states not occupied by other agents
        eligible_states_id = np.nonzero(T_eligible[:, prev_loc_1d])[
            0
        ]  # * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
        value_eligible = value[
            eligible_states_id
        ].flatten()  # value of eligible states plus some noise

        # ACTION SELECTION

        if doProbabilisticPolicy:
            if doSoftmaxPolicy:
                prob_arr = util.softmax(
                    value_eligible, temp=exploration_bias
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



        # ------ Locations of agents and rewards at each time point --------
        xloc_prev, yloc_prev = util.loc1Dto2D(prev_loc_1d, edge_size)
        xloc_next, yloc_next = util.loc1Dto2D(next_loc_1d, edge_size)
        
        x_agents_all[ai, ti] = xloc_prev
        y_agents_all[ai, ti] = yloc_prev
        
        # ------- compute energy cost of moving to new location --------------
        dist_traveled = np.sqrt(
            (xloc_next - xloc_prev) ** 2 + (yloc_next - yloc_prev) ** 2
        )
        calories_expended_mat[ai, ti] = (
            caloric_cost_per_unit_dist * dist_traveled
        )
        agent.energy_total -= calories_expended_mat[ai, ti]

        # ------------- compute metrics for data analysis -----------------
        if len(list_agents) > 1:
            dist_to_neighbors = np.sqrt(
                (xloc_otheragents - xloc_self) ** 2
                + (yloc_otheragents - yloc_self) ** 2
            )
            dist_to_nearest_neighbor_all[ai, ti] = np.min(
                dist_to_neighbors
            )

        calories_acquired_all[ai, ti] = calories_acquired_mat[
            ai, ti
        ]

        if phi_food[next_loc_1d][0]:
            agent.times_at_food.append(
                ti + 1
            )  # add this frame to the list of frames where agent is at a food location

        # -------------------------------------------------------------------

        agent.state_trajectory[ti + 1] = next_loc_1d  # scalar
        agent.value_trajectory[
            :, ti + 1
        ] = value.flatten()  # (N_states, N_timesteps)
        agent.energy_trajectory[ti + 1] = agent.energy_total
        calories_total_mat[ai, ti + 1] = (
            calories_total_mat[ai, ti]
            + calories_acquired_mat[ai, ti]
            - calories_expended_mat[ai, ti]
        )

        phi_agents[next_loc_1d] += 1  # move into new location
        loc_1d_allagents[ai] = next_loc_1d

for ai, agent in enumerate(list_agents):
    # print(agent.times_at_food)
    if len(agent.times_at_food) == 0:
        time_to_first_food_all[ai] = N_timesteps  # TO DO: figure out how to track agents who never reach food
    else:
        time_to_first_food_all[ai] = agent.times_at_food[
            0
        ]  # fetch the first time step where agent is at a food location

endtime = time.time()

if doPrintAgentStateTrajectories:
    for ai, agent in enumerate(list_agents):
        print(agent.state_trajectory)

print("simulation run time = " + str(endtime - starttime))

# -------Save locations of each agent and reward in data frames-----------
# all_birdsDF: 
# all_rewardsDF:

# Bird locations  
birds_all = []
for ai in range(N_agents):
    single_agent = pd.DataFrame(
        {
            "x": x_agents_all[ai, :],
            "y": y_agents_all[ai, :],
            "time": range(1, N_timesteps + 1),
            "bird": ai + 1,
            "type": "random",
        }
    )

    birds_all.append(single_agent)
    
all_birdsDF = pd.concat(birds_all)

all_birdsDF.head()

# Reward locations 
rewards_all = []
for ti in range(N_timesteps):
    loc1D = [idx for idx, val in enumerate(food_trajectory[:, ti]) if val > 0]
    if len(loc1D) > 0:
        x_reward, y_reward = util.loc1Dto2D(loc1D, edge_size)
        single_time = pd.DataFrame(
            {
                "x": x_reward,
                "y": y_reward,
                "time": np.repeat(ti, len(x_reward)),
            }
        )
        rewards_all.append(single_time)
    
all_rewardsDF = pd.concat(rewards_all)

print(all_birdsDF)
print(all_rewardsDF)
    
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)

# %%
# ************************* CREATE USER INTERFACE ***************************

if doAnimation:
    fig_ui, (ax_main, ax_value) = plt.subplots(1, 2, figsize=(10, 4))

    ax_main.set_title("Environment")
    # phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])
    min_cal = -0.1
    max_cal = np.max(food_trajectory)
    sns.heatmap(
        ax=ax_main,
        data=food_init_loc_2d,
        vmin=min_cal,
        vmax=max_cal,
        center=0,
        square=True,
        cbar=True,
    )

    # Choose one agent to highlight in magenta. Plot in an adjacent
    # panel the value function of this agent evolving over time
    z = 0  # index of highlighted agent
    value_heatmap = np.reshape(
        list_agents[z].value_trajectory[:, 0], [edge_size, edge_size]
    )
    # scale the color bar based on the c weights (TO DO: find a better way to do this )
    vmin = np.min(np.min(c_weights), 0)
    vmax = np.max(np.max(c_weights), 0)
    min_value = 0
    max_value = 1
    sns.heatmap(
        ax=ax_value,
        data=value_heatmap,
        vmin=min_value,
        vmax=max_value,
        center=0,
        square=True,
        cbar=True,
    )
    ax_value.set_title("Value function for one agent")

    list_plot_agents = []
    markers = ["co"]  # color for all other agents

    # plot each agent's data
    for ai, agent in enumerate(list_agents):
        if ai == z:
            marker = "mo"
        else:
            marker = markers[0]
        (one_plot,) = ax_main.plot([], [], marker, markersize=3)
        list_plot_agents.append(one_plot)

    def update_plot(
        frame, list_agents, food_trajectory, edge_size
    ):  # , list_food_loc_id, edge_size):
        fig_ui.suptitle("frame " + str(frame) + " / " + str(N_timesteps))
        # food locations and quantities
<<<<<<< HEAD
        food_heatmap = np.reshape(food_trajectory[:, frame], [edge_size, edge_size])
=======
        food_heatmap = np.reshape(
            food_trajectory[:, frame], [edge_size, edge_size]
        )
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
        sns.heatmap(
            ax=ax_main,
            data=food_heatmap,
            vmin=min_cal,
            vmax=max_cal,
            center=0,
            square=True,
            cbar=False,
        )  # , cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})

        # x_food, y_food = util.loc1Dto2D(list_food_loc_id, edge_size)
        # plot_food.set_data(x_food, y_food)

        # state of each agent
        for ai, agent in enumerate(list_agents):
            state = int(agent.state_trajectory[frame])
            x, y = util.loc1Dto2D(state, edge_size)
            list_plot_agents[ai].set_data(x, y)

        # value function of one agent
        ax_value.cla()
        value_heatmap = np.reshape(
            list_agents[z].value_trajectory[:, frame], [edge_size, edge_size]
        )
        sns.heatmap(
            ax=ax_value,
            data=value_heatmap,
            vmin=min_value,
            vmax=max_value,
            center=0,
            square=True,
            cbar=False,
        )  # , cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})

        return list_plot_agents  # , plot_food

    fig_ui.tight_layout()

    ani = animation.FuncAnimation(
        fig_ui,
        update_plot,
        frames=N_timesteps,
        fargs=(list_agents, food_trajectory, edge_size),
    )  # , list_food_loc_id, edge_size))


# %% **********************  SAVE DATA ****************************************
# saveData = False

# filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\simulated_data\sim1"
filepath_data = directory_data + filename_data
file_data = Path(filepath_data + ".sav")
# saving to m4 using ffmpeg writer
if file_data.exists():
    filepath_data = filepath_data + "_v2"

# save the model to disk
# filename = filepath + '.sav'
dictionary = {
    "N_timesteps": N_timesteps,
    "N_agents": N_agents,
    "N_food_units_total": N_food_units_total,
    "patch_dim": patch_dim,
    "example_agent": agent,
    "sight_radius": sight_radius,
    "discount_factor": discount_factor,
    "c_weights": c_weights,
}

pickle.dump(dictionary, open(filepath_data + ".sav", "wb"))


# %% **********************  SAVE ANIMATION  ****************************************

# saveMovie = False

if saveMovie:
    # filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_foodonly_v2.gif"
    # filepath = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_simple_explorebias10"

    filepath_movie = directory_movie + filename_movie
    moviefile = Path(filepath_movie + ".gif")
    # saving to m4 using ffmpeg writer
    if moviefile.exists():
        filepath_movie = filepath_movie + "_v2"

<<<<<<< HEAD
    ani.save(filepath_movie + ".gif", dpi=300, writer=animation.PillowWriter(fps=2))

    # ## saving as an mp4
    # filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_mp4.mp4"
    # ani.save(filename, writer=animation.FFMpegWriter(fps=30) )
=======
    ani.save(
        filepath_movie + ".gif", dpi=300, writer=animation.PillowWriter(fps=2)
    )

    # ## saving as an mp4
    # filename = r"C:\Users\admin\Dropbox\Code\Basis-code\multiagent_mp4.mp4"
    # ani.save(filename, writer=animation.FFMpegWriter(fps=30) )
>>>>>>> 1e29dff (script that runs a single simulation of communicator birds and outputs dataframes storing locations of birds and rewards)
