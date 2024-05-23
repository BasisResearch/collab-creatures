# %%NBQA-CELL-SEPf56012
import logging
import os
from itertools import product

import numpy as np
import pandas as pd

import collab.foraging.communicators as com
import collab.foraging.toolkit as ft
from collab.utils import find_repo_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# if you need to change the number of frames, replace 50 with your desired number

# this is an alternative continuous development setup
# for automated notebook testing
# feel free to ignore this
dev_mode = False  # set to True if you want to generate your own csvs
smoke_test = "CI" in os.environ
N_frames = 10 if smoke_test else 120


# %%NBQA-CELL-SEPf56012
# Simulation setup 1 for the communication detection problem
from collab.utils import find_repo_root

repo_root = find_repo_root()
home_dir = os.path.join(repo_root, "data/foraging/communicators/communicators_strong/")
# # agent parameters
sight_radius = [6]
c_trust = [0, 0.6]  # 0: ignorers
N_agents = 9

# # environment parameters
edge_size = 30
N_total_food_units = 24
reward_patch_dim = [4]  # clustered is 4, distributed is 1

# simulation parameters
N_runs = 1  # How many times would you like to run each case?
N_frames = N_frames

# Generate a dataframe containing all possible combinations of the parameter values specified above.
param_list = [i for i in product(c_trust, sight_radius, reward_patch_dim)]
metadataDF = pd.DataFrame(param_list)
metadataDF.columns = ["c_trust", "sight_radius", "reward_patch_dim"]
metadataDF["sim index"] = np.arange(len(metadataDF)).astype(int)
N_sims = len(metadataDF) if not smoke_test else 1

# save metadata to home directory
if dev_mode and not smoke_test:
    metadataDF.to_csv(os.path.join(home_dir, "metadataDF.csv"))
    pd.DataFrame(
        [
            {
                "N_sims": N_sims,
                "N_runs": N_runs,
                "N_frames": N_frames,
                "N_agents": N_agents,
                "N_total_food_units": N_total_food_units,
                "edge_size": edge_size,
            }
        ]
    ).to_csv(os.path.join(home_dir, "additional_meta_params.csv"))

display(metadataDF)

print(metadataDF.shape)

# Simulations set right,
# Before you start, keep in sight,
# Data safe from overwrite.
print(home_dir)


# %%NBQA-CELL-SEPf56012
def run_simulations(fresh_start=True):
    if fresh_start:
        start = 0
    else:
        resultsDF = pd.read_csv(os.path.join(home_dir, "resultsDF.csv"))
        start = resultsDF.iloc[-1]["sim index"].astype(
            int
        )  # start with the last existing batch

        logging.info(f"Starting from batch {start+1}.")

    all_results = []

    for si in range(start, N_sims):
        # 1. pull out parameters from row si in the metadata
        df_row = metadataDF.iloc[[si]]
        c_trust = df_row["c_trust"].iloc[0]
        sight_radius = df_row["sight_radius"].iloc[0]
        reward_patch_dim = df_row["reward_patch_dim"].iloc[0].astype(int)

        # arrays to save success measures for each run of this simulation
        mean_times_to_first_reward = np.zeros((N_runs))
        num_foragers_failed = np.zeros((N_runs))

        logging.info(
            f"Starting simulation setting {si+1}/{N_sims}, about to run it {N_runs} times."
        )

        # Do multiple runs of the simulation and store the results in a results dataframe
        batch_results = []
        for ri in range(N_runs):
            # initialize environment
            env = com.Environment(
                edge_size=edge_size,
                N_total_food_units=N_total_food_units,
                patch_dim=reward_patch_dim,
            )
            env.add_food_patches()

            # run simulation
            sim = com.SimulateCommunicators(
                env, N_frames, N_agents, c_trust=c_trust, sight_radius=sight_radius
            )
            sim.run()

            # Compute success measures
            time_to_first_allforagers = np.zeros(N_agents)
            for forager_id in range(
                1, N_agents + 1
            ):  # compute time to first food for each forager
                singleforagerDF = sim.all_foragersDF.loc[
                    sim.all_foragersDF.forager == forager_id
                ]
                time_to_first_allforagers[forager_id - 1] = (
                    com.compute_time_to_first_reward(
                        singleforagerDF, sim.all_rewardsDF, N_frames
                    )
                )
            mean_times_to_first_reward = np.mean(
                time_to_first_allforagers
            )  # take the average across foragers
            num_foragers_failed = np.sum(
                time_to_first_allforagers == N_frames
            )  # number of foragers that failed to reach food

            # Save the simulation results in a folder named sim{si}_run{ri} in the home directory
            sim_folder = "sim" + str(si) + "_run" + str(ri)
            sim_dir = os.path.join(home_dir, sim_folder)
            if not os.path.isdir(sim_dir):
                os.makedirs(sim_dir)
            if dev_mode and not smoke_test:
                sim.all_foragersDF.to_csv(os.path.join(sim_dir, "foragerlocsDF.csv"))
                sim.all_rewardsDF.to_csv(os.path.join(sim_dir, "rewardlocsDF.csv"))

            # Combine the metadata and the success measures for the results dataframe
            results_onesim = {
                "c_trust": c_trust,
                "sight_radius": sight_radius,
                "reward_patch_dim": reward_patch_dim,
                "sim index": si,
                "run index": ri,
                "time to first food": mean_times_to_first_reward,
                "num foragers failed": num_foragers_failed,
            }
            batch_results.append(results_onesim)

        batch_resultsDF = pd.DataFrame(batch_results)

        if "resultsDF" in locals():
            resultsDF = pd.concat(
                [resultsDF, batch_resultsDF], ignore_index=True, axis=0
            )
        else:
            resultsDF = batch_resultsDF.copy()

        if dev_mode and not smoke_test:
            resultsDF.to_csv(os.path.join(home_dir, "resultsDF.csv"))
            logging.info(f"Saved results for batch {si+1}/{N_sims}.")


run_simulations(fresh_start=True)


# %%NBQA-CELL-SEPf56012
# to make sure the results make sense
# feel free to animate one of them

# load the data from the first simulation
sim_folder = "sim0_run0"
sim_dir = os.path.join(home_dir, sim_folder)
foragerlocsDF = pd.read_csv(os.path.join(sim_dir, "foragerlocsDF.csv"), index_col=0)
rewardlocsDF = pd.read_csv(os.path.join(sim_dir, "rewardlocsDF.csv"), index_col=0)
communicators = ft.object_from_data(foragerlocsDF, rewardsDF=rewardlocsDF)


# %%NBQA-CELL-SEPf56012
# animate

ft.animate_foragers(
    communicators, plot_rewards=True, width=600, height=600, point_size=8
)


# %%NBQA-CELL-SEPf56012
# custom list of locations for simulation setup 2
# with focus on low values of c_trust
# the main simulation run is commented out
# uncomment if you want to run many of simulations
min_value = 0.0
max_value = 0.7
density1 = 0.005
density2 = 0.01
c_locations = []

current_value = min_value
while current_value < 0.3:
    c_locations.append(current_value)
    current_value += density1
while current_value <= max_value:
    c_locations.append(current_value)
    current_value += density2

# # Simulation setup 2 for the impact of communication
home_dir = os.path.join(repo_root, "data/foraging/communicators/communicators_weak/")
# agent parameters
sight_radius = [5]
c_trust = c_locations
# 0: ignorers,
N_agents = 9

# environment parameters
edge_size = 45
N_total_food_units = 16
reward_patch_dim = [1, 2, 4]  # clustered is 4, distributed is 1

# simulation parameters
N_runs = 2  # How many times would you like to run each case?
N_frames = N_frames

# Generate a dataframe containing all possible combinations of the parameter values specified above.
param_list = [i for i in product(c_trust, sight_radius, reward_patch_dim)]
metadataDF = pd.DataFrame(param_list)
metadataDF.columns = ["c_trust", "sight_radius", "reward_patch_dim"]
metadataDF["sim index"] = np.arange(len(metadataDF)).astype(int)
N_sims = len(metadataDF)

# save metadata to home directory
if dev_mode and not smoke_test:
    metadataDF.to_csv(os.path.join(home_dir, "metadataDF.csv"))
    pd.DataFrame(
        [
            {
                "N_sims": N_sims,
                "N_runs": N_runs,
                "N_frames": N_frames,
                "N_agents": N_agents,
                "N_total_food_units": N_total_food_units,
                "edge_size": edge_size,
            }
        ]
    ).to_csv(os.path.join(home_dir, "additional_meta_params.csv"))

display(metadataDF)

print(home_dir)

# uncomment if you want to run 600 simulations
# and turn on dev_mode if you want to overwrite the csvs
# run_simulations(fresh_start = True)
