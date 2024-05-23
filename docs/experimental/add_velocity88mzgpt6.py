# %%NBQA-CELL-SEP967117
# importing packages. See https://github.com/BasisResearch/collab-creatures for repo setup
import logging
import os
import random
import time
import warnings

import dill
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np

from collab.utils import find_repo_root

root = find_repo_root()
from collab.foraging import random_hungry_followers as rhf
from collab.foraging.toolkit.velocity import add_velocities_to_foragers

logging.basicConfig(format="%(message)s", level=logging.INFO)

# users can ignore smoke_test -- it's for automatic testing on GitHub, to make sure the notebook runs on future updates to the repository
smoke_test = "CI" in os.environ
num_frames = 5 if smoke_test else 50
num_svi_iters = 10 if smoke_test else 1000
num_samples = 10 if smoke_test else 1000


notebook_starts = time.time()

# %%NBQA-CELL-SEP967117
random.seed(23)
np.random.seed(23)

random_foragers_sim = rhf.RandomForagers(
    grid_size=40,
    probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
    num_foragers=3,
    num_frames=num_frames,
    num_rewards=15,
    grab_range=3,
)

random_foragers_sim()

random_foragers_sim.foragersDF.head()

foragers = random_foragers_sim.foragers
len(foragers)

add_velocities_to_foragers(foragers)

display(foragers[0])

# %%NBQA-CELL-SEP967117
sampling_rate = 0.01

path = os.path.join(
    root,
    f"data/foraging/central_park_birds_cleaned_2022/central_park_objects_sampling_rate_{sampling_rate}.pkl",
)

with open(path, "rb") as file:
    central_park_objects = dill.load(file)

ducks_objects = central_park_objects[0]
ducks_50 = ducks_objects[50]
add_velocities_to_foragers(ducks_50.foragers)

display(ducks_50.foragers[5])

# %%NBQA-CELL-SEP967117
# sanity check, plot velocities against time
# unexpectedly large values should be investigated

for forager in ducks_50.foragers:

    df = forager

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for forager, group in df.groupby("forager"):
        plt.plot(
            group["time"], group["velocity_x"], marker="o", label=f"Forager {forager}"
        )
    plt.xlabel("Time")
    plt.ylabel("Velocity X")
    plt.title("Velocity X vs Time")
    plt.legend()
    plt.grid(True)

    # Plot velocity_y vs. time
    plt.subplot(1, 2, 2)
    for forager, group in df.groupby("forager"):
        plt.plot(
            group["time"], group["velocity_y"], marker="o", label=f"Forager {forager}"
        )
    plt.xlabel("Time")
    plt.ylabel("Velocity Y")
    plt.title("Velocity Y vs Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
