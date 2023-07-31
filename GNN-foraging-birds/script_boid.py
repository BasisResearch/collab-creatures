import GNN_foraging_birds 
import numpy as np
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU, Module
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import csv
import os
from datetime import datetime

# make folder for gif outputs
now = datetime.now()
now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
plots_dir = '/Users/emily/Downloads/plots_tmp'
plots_dir = os.path.join(plots_dir, now_str)
os.makedirs(plots_dir, exist_ok=True)

# Boids simulation parameters
num_birds = 25
num_steps_train = 150
num_sims_train = 150
num_steps_test = 300
grid_size = 40
neighbor_radius = 8
close_radius = 5
alignment_weight = 0.1
cohesion_weight = 0.05
separation_weight = 0.1
max_speed=.5

# # Initial bird positions and velocities
positions = np.random.rand(num_birds, 2) * grid_size
velocities = (np.random.rand(num_birds, 2) - 0.5) * 2

# define generative model 
def boid_model(positions, velocities):
    _, new_velocities = GNN_foraging_birds.step_position_velocity_boid(positions, velocities, neighbor_radius, close_radius, alignment_weight, cohesion_weight, separation_weight, grid_size, max_speed)
    return new_velocities

# # Generate actual trajectories for plotting
# actual_trajectories = GNN_foraging_birds.generate_trajectories(boid_model, positions.copy(), velocities.copy(), num_steps_test, neighbor_radius=neighbor_radius, grid_size=grid_size)

# # Plot actual and predicted trajectories
# GNN_foraging_birds.animate_agents(  actual_trajectories=actual_trajectories, 
#                                     num_steps_test=num_steps_test, 
#                                     savename = 'output_boid_trajectories.gif', 
#                                     grid_size=grid_size)

# Generate training dataset
dataset = GNN_foraging_birds.generate_dataset_boid(num_birds, num_sims_train, num_steps_train, neighbor_radius, close_radius, alignment_weight, cohesion_weight, separation_weight, grid_size, max_speed)

# Train GNN model
for meta_epoch in range(20):
    print('Meta epoch: ', meta_epoch)
    model = GNN_foraging_birds.train_GNN_model(dataset, hidden_dim=32, num_epochs=20, learning_rate=0.002, load_path='gnn_extended_params_boid.pth', save_path='gnn_extended_params_boid.pth')
    torch.save(model.state_dict(), os.path.join(plots_dir, 'gnn_extended_params_boid_meta_epoch'+str(meta_epoch)+'.pth'))

    # Initial bird positions and velocities
    positions = np.random.rand(num_birds, 2) * grid_size
    velocities = (np.random.rand(num_birds, 2) - 0.5) * 2

    # Generate actual and predicted trajectories
    actual_trajectories = GNN_foraging_birds.generate_trajectories(boid_model, positions.copy(), velocities.copy(), num_steps_test, neighbor_radius=neighbor_radius, grid_size=grid_size)
    predicted_trajectories = GNN_foraging_birds.generate_trajectories(model, positions.copy(), velocities.copy(), num_steps_test, neighbor_radius=neighbor_radius, grid_size=grid_size)

    # Plot actual and predicted trajectories
    GNN_foraging_birds.animate_agents(  actual_trajectories=actual_trajectories, 
                                        predicted_trajectories=predicted_trajectories, 
                                        num_steps_test=num_steps_test, 
                                        savename = os.path.join(plots_dir, 'trajectories_meta_epoch'+str(meta_epoch)+'.gif'), 
                                        grid_size=grid_size)
