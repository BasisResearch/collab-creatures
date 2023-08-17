# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:07:08 2023

@author: admin
"""
# import sys
# sys.path.insert(0, "..")
import simulation 
import environments 
import agents
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
<<<<<<< HEAD
# import figures 
import utils as util
=======
import figures 
import gridworld_utils as util
>>>>>>> e15390c (version 1 of refactoring simulation sandbox)
from importlib import reload
import time
import imp 
reload(util)
<<<<<<< HEAD
# reload(figures)
=======
reload(figures)
>>>>>>> e15390c (version 1 of refactoring simulation sandbox)
reload(environments)
reload(agents)
reload(simulation)

doAnimation = False

# Create an initialize a new environment 
env = environments.Environment(edge_size=30, N_total_food_units=16, patch_dim=1, max_step_size=3)

# Add food  --> may want to put this inside the Simulation class 
food_statistics_type = "drop_food_once"
env.add_food_patches(food_statistics_type=food_statistics_type)

plt.figure()
plt.imshow(np.reshape(env.food_calories_by_loc, (env.edge_size, env.edge_size)))

# Create a new simulation with that environment and add agents into the simulation 
# want to add in it
N_frames = 3
sim = simulation.Simulation(env, N_agents=2, N_frames=N_frames)
sim.run()
print(sim.all_birdsDF)
print(sim.all_rewardsDF) 



# c_food_self = 0.9
# c_food_others = 0.1  # to what extent do the birds care about information from other birds?
# c_otheragents = 0
# c_group = 0

# N_frames = 3

# bird = TreeWorld.BirdAgent(env, N_frames)
# value_update = bird.value_update(c_weights, reward_vec_list)

# if doAnimation:
#     fig_ui, (ax_main, ax_value) = plt.subplots(1, 2, figsize=(10, 4))

#     ax_main.set_title("Environment")
#     # phi_food_2d = np.reshape(phi_food, [edge_size,edge_size])
#     min_cal = -0.1
#     max_cal = np.max(sim.food_trajectory)
#     sns.heatmap(
#         ax=ax_main,
#         data=env.phi_food_init,
#         vmin=min_cal,
#         vmax=max_cal,
#         center=0,
#         square=True,
#         cbar=True,
#     )

#     # Choose one agent to highlight in magenta. Plot in an adjacent
#     # panel the value function of this agent evolving over time
#     z = 0  # index of highlighted agent
#     value_heatmap = np.reshape(
#         sim.list_agents[z].value_trajectory[:, 0], [env.edge_size, env.edge_size]
#     )
#     # scale the color bar based on the c weights (TO DO: find a better way to do this )
#     vmin = 0 #np.min(np.min(sim.weights_list), 0)
#     vmax = 1 #np.max(np.max(c_weights), 0)
#     min_value = 0
#     max_value = 1
#     sns.heatmap(
#         ax=ax_value,
#         data=value_heatmap,
#         vmin=min_value,
#         vmax=max_value,
#         center=0,
#         square=True,
#         cbar=True,
#     )
#     ax_value.set_title("Value function for one agent")

#     list_plot_agents = []
#     markers = ["co"]  # color for all other agents

#     # plot each agent's data
#     for ai, agent in enumerate(sim.list_agents):
#         if ai == z:
#             marker = "mo"
#         else:
#             marker = markers[0]
#         (one_plot,) = ax_main.plot([], [], marker, markersize=3)
#         list_plot_agents.append(one_plot)

#     def update_plot(
#         frame, list_agents, food_trajectory, edge_size
#     ):  # , list_food_loc_id, edge_size):
#         fig_ui.suptitle("frame " + str(frame) + " / " + str(N_frames))
#         # food locations and quantities
#         food_heatmap = np.reshape(
#             food_trajectory[:, frame], [edge_size, edge_size]
#         )
#         sns.heatmap(
#             ax=ax_main,
#             data=food_heatmap,
#             vmin=min_cal,
#             vmax=max_cal,
#             center=0,
#             square=True,
#             cbar=False,
#         )  # , cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})

#         # x_food, y_food = util.loc1Dto2D(list_food_loc_id, edge_size)
#         # plot_food.set_data(x_food, y_food)

#         # state of each agent
#         for ai, agent in enumerate(sim.list_agents):
#             state = int(agent.state_trajectory[frame])
#             x, y = util.loc1Dto2D(state, edge_size)
#             list_plot_agents[ai].set_data(x, y)

#         # value function of one agent
#         ax_value.cla()
#         value_heatmap = np.reshape(
#             sim.list_agents[z].value_trajectory[:, frame], [edge_size, edge_size]
#         )
#         sns.heatmap(
#             ax=ax_value,
#             data=value_heatmap,
#             vmin=min_value,
#             vmax=max_value,
#             center=0,
#             square=True,
#             cbar=False,
#         )  # , cbar_ax=ax_cbar, cbar_kws={'shrink':0.5})

#         return list_plot_agents  # , plot_food

#     fig_ui.tight_layout()

#     ani = animation.FuncAnimation(
#         fig_ui,
#         update_plot,
#         frames=N_frames,
#         fargs=(sim.list_agents, sim.food_trajectory, env.edge_size),
#     )  # , list_food_loc_id, edge_size))
