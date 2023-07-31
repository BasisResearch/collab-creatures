import numpy as np
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU, Module
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import csv
from torch_geometric.nn import BatchNorm


def generate_dataset_boid(num_birds, num_sims_train, num_steps_train, neighbor_radius=15, close_radius=3, alignment_weight=0.1, cohesion_weight=0.1, separation_weight=0.1, grid_size=30):
    dataset = []

    for sim in range(num_sims_train):
        # Reset bird positions and velocities
        positions = np.random.rand(num_birds, 2) * grid_size
        velocities = (np.random.rand(num_birds, 2) - 0.5) * 2  # random velocities between -1 and 1
        for step in range(num_steps_train):
            new_positions, new_velocities = step_position_velocity_boid(positions, velocities, neighbor_radius, close_radius, alignment_weight, cohesion_weight, separation_weight, grid_size)

            # Compute edges (i.e., which birds are neighbors)
            dists = np.sqrt(((positions[:, None] - positions)**2).sum(-1))
            edges = np.argwhere(dists < neighbor_radius)

            # Convert to PyTorch tensors
            x = torch.tensor(np.hstack([positions, velocities]), dtype=torch.float)
            edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
            y = torch.tensor(new_velocities, dtype=torch.float)

            # Create a PyTorch Geometric Data object and add it to the dataset
            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)

            # Update positions and velocities for next step
            positions = new_positions
            velocities = new_velocities
            
    return dataset

def step_position_velocity_boid(positions, velocities, neighbor_radius=15, close_radius=3, alignment_weight=0.01, cohesion_weight=0.01, separation_weight=0.01, grid_size=30):
    num_birds = len(positions)

    # Compute pairwise distances
    dists = np.sqrt(((positions[:, None] - positions)**2).sum(-1))

    # Compute mask of neighbor birds and close birds
    neighbors = dists < neighbor_radius
    close_birds = dists < close_radius

    new_velocities = np.empty_like(velocities)

    for i in range(num_birds):
        # Check if there are no neighbors
        if not neighbors[i].any():
            new_velocities[i] = velocities[i]
            continue

        # Alignment: Match velocity to nearby flockmates
        avg_neighbor_velocity = velocities[neighbors[i]].mean(axis=0)
        alignment = avg_neighbor_velocity - velocities[i]

        # Cohesion: Steer towards average position of nearby flockmates
        avg_neighbor_position = positions[neighbors[i]].mean(axis=0)
        cohesion = avg_neighbor_position - positions[i]

        # Separation: Avoid getting too close to flockmates
        if not close_birds[i].any():
            new_velocities[i] = velocities[i]
            continue
        separation_vectors = positions[i] - positions[close_birds[i]]
        separation_distances = dists[i,close_birds[i]]
        # Normalize each separation vector by its distance
        separation_vectors_normalized = separation_vectors / np.maximum(separation_distances[:, None], .01)
        # Sum the normalized vectors to get the final separation vector
        separation = separation_vectors_normalized.sum(axis=0)

        # Compute new velocity
        new_velocities[i] = velocities[i] + alignment_weight * alignment + cohesion_weight * cohesion + separation_weight * separation

    # Limit speed
    speeds = np.sqrt((new_velocities**2).sum(axis=1))
    over_speed_birds = speeds > 1
    new_velocities[over_speed_birds] = new_velocities[over_speed_birds] / speeds[over_speed_birds, None]

    # Update positions
    new_positions = positions + new_velocities

    # Bouncing boundary conditions
    # Create a mask for birds that are out of bounds
    mask_outside_x = (new_positions[:, 0] < 0) | (new_positions[:, 0] > grid_size)
    mask_outside_y = (new_positions[:, 1] < 0) | (new_positions[:, 1] > grid_size)

    # Bounce back if position is outside grid
    new_positions[mask_outside_x, 0] = np.clip(new_positions[mask_outside_x, 0], 0, grid_size)
    new_positions[mask_outside_y, 1] = np.clip(new_positions[mask_outside_y, 1], 0, grid_size)

    # Reverse velocity
    new_velocities[mask_outside_x, 0] *= -1
    new_velocities[mask_outside_y, 1] *= -1

    return new_positions, new_velocities


class GNN_foraging_simple(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_foraging_simple, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = ReLU()(x)
        x = self.conv2(x, edge_index)
        x = ReLU()(x)
        x = self.fc(x)
        return x


class GNN_foraging_complex(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_foraging_complex, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.fc(x)
        return x

class GNN_foraging_extended(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_foraging_extended, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Additional GCN layer
        self.bn3 = BatchNorm(hidden_dim)  # Additional BatchNorm layer
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))  # Use the additional GCN layer
        x = self.bn3(x)  # Use the additional BatchNorm layer
        x = self.fc(x)
        return x


def train_GNN_model(dataset, hidden_dim=32, num_epochs=100, learning_rate=0.01, load_path=None, save_path=None):
    # inputs are positions and velocities, outputs are new velocities
    # Initialize the model and the optimizer
    model = GNN_foraging_extended(input_dim=4, hidden_dim=hidden_dim, output_dim=2)  # 4 inputs (x, y, vx, vy), 2 outputs (vx, vy)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # If a path is provided, load model parameters
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    # Create a DataLoader for batching the dataset
    loader = DataLoader(dataset, batch_size=32)

    # List to store losses
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = ((out - data.y)**2).mean()  # Mean Squared Error loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(loader)
        print(f'Epoch: {epoch}, Loss: {epoch_loss}')
        losses.append(epoch_loss)

    # If a path is provided, save model parameters
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    # Save losses to CSV file
    with open('output_losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        for i, loss in enumerate(losses):
            writer.writerow([i, loss])

    # Plot losses and save as an image
    plt.figure()
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('output_loss_plot.png')
        
    return model


def generate_trajectories(model, positions, velocities, num_steps, neighbor_radius=15, grid_size=30):
    # Store trajectories for each bird
    trajectories = np.empty((num_steps, len(positions), 2))
    trajectories[0] = positions

    for step in range(1, num_steps):
        
        if isinstance(model, Module):
            # Model is a PyTorch module

            # Convert to PyTorch tensors
            positions_t = torch.tensor(positions, dtype=torch.float)
            velocities_t = torch.tensor(velocities, dtype=torch.float)
            x = torch.cat([positions_t, velocities_t], dim=-1)

            # Compute pairwise distances and find neighbors
            dists = np.sqrt(((positions[:, None] - positions)**2).sum(-1))
            edges = np.argwhere(dists < neighbor_radius)
            edges = edges[edges[:, 0] != edges[:, 1]]
            edge_index = torch.tensor(edges.transpose(), dtype=torch.long)

            new_velocities = model(x, edge_index).detach().numpy()
        else:
            # Model is a function like compute_new_velocities
            new_velocities = model(positions, velocities)

        # Update positions and velocities
        positions += velocities
        velocities = new_velocities

        # Store new positions
        trajectories[step] = positions

    return trajectories


def animate_agents(actual_trajectories, predicted_trajectories=[], num_steps_test=10, savename = 'trajectories.gif', grid_size=30):
    if predicted_trajectories == []:
        predicted_trajectories = actual_trajectories

    # Gif animation to plot the trajectories
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.xticks([])
    # ax.set_title('Actual and Predicted Trajectories')

    fade_steps = 10
    points_actual = []
    points_predicted = []

    # init function for animation
    def init():
        return ax,

    def animate(i):
        # plot new positions
        points_actual.append(ax.scatter(actual_trajectories[i, :, 0], actual_trajectories[i, :, 1], color='black', label='Actual' if i==0 else "_nolegend_"))
        points_predicted.append(ax.scatter(predicted_trajectories[i, :, 0], predicted_trajectories[i, :, 1], color='red', label='Predicted' if i==0 else "_nolegend_"))   
        
        # fade out old positions
        for points in [points_actual, points_predicted]:
            for pi in points:
                current_alpha = pi.get_alpha()
                if current_alpha is None:
                    current_alpha = 1.0
                new_alpha = current_alpha / 2
                pi.set_alpha(new_alpha)
                if new_alpha < 0.05:
                    pi.remove()
                    points.remove(pi)


        return ax,

    # Create the animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_steps_test, interval=50)

    # Save the animation
    anim.save(savename, writer='imagemagick')