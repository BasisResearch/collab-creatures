import logging

import numpy as np
import pandas as pd
import copy 
from collab.foraging.random_hungry_followers.rhf_helpers import (
    construct_visibility,
    generate_proximity_score,
    update_rewards,
)
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")


def add_follower_foragers(
    sim,
    num_follower_foragers=3,
    visibility_range=10,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    initial_positions=None,
):
    """
        A function to add follower foragers to a simulation.

    Args:
        sim (foragers): A foragers object.

    Returns:
        sim (foragers): The same foragers object, but with follower foragers added.
    """

    old_foragers = sim.foragers.copy()

    how_many_foragers_already = len(old_foragers)

    new_foragers = sim.generate_random_foragers(
        num_follower_foragers, size=1, initial_positions=initial_positions
    )["random_foragers"]

    for new_forager in new_foragers:
        new_forager["forager"] = new_forager["forager"] + how_many_foragers_already
        new_forager["type"] = "follower"

    for t in range(0, sim.num_frames):

        sim.foragers.extend(new_foragers)


        if t > 0 and t % 10 == 0:
            logging.info(f"Generating frame {t}/{sim.num_frames} ")

        _vis = construct_visibility(
            new_foragers,
            sim.grid_size,
            visibility_range=visibility_range,
            start=t,
            end=t + 1,
        )["visibility"]


        _prox = generate_proximity_score(
            new_foragers,
            _vis,
            visibility_range=visibility_range,
            getting_worse=getting_worse,
            optimal=optimal,
            proximity_decay=proximity_decay,
            start=0, # these are relative positions to _vis, 
                     # which is generated for t only, so 0 in _vis is t
            end=1,
            time_shift=t,
        )["proximity"]


        for b in range(num_follower_foragers):
            options = copy.deepcopy(_vis[b][0])

            
            options = options.merge(_prox[b][0], how="inner")

            if b == 1:
                plt.scatter(options["x"], options["y"], c = options["proximity"])
                plt.colorbar()
                plt.title(f"Visibility for follower {b} at time {t}")
                plt.xlim(0, 60)
                plt.ylim(0, 60)
                plt.show()


            options.sort_values(by="proximity", ascending=False, inplace=True)
            options = options.head(10)
            if b == 1:
                print("options", options)

            chosen_option = options.iloc[np.random.randint(0, 10)]

            if t < sim.num_frames - 1:
                new_x = chosen_option["x"]
                new_y = chosen_option["y"]



                new_row = {
                    "x": new_x,
                    "y": new_y,
                    "time": t + 1,
                    "forager": b,
                    "type": "follower",
                }
               
                new_foragers[b] = pd.DataFrame(new_row, index = [0])
                #new_foragers[b].loc[len(new_foragers[b])] = new_row


    sim.foragersDF = pd.concat(sim.foragers)
                   
    

    rew = update_rewards(sim, sim.rewards, sim.foragers, start=1)

    sim.rewards = rew["rewards"]
    sim.rewardsDF = rew["rewardsDF"]

    return sim



import collab.foraging.toolkit as ft
from collab.foraging.random_hungry_followers.random_foragers import Foragers
import random

num_frames = 5


random.seed(23)
np.random.seed(23)

# create a new empty simulation (a starting point for the actual simulation)
grid_size = 60
follower_sim = Foragers(
    grid_size=grid_size,
    num_foragers=3,
    num_frames=num_frames,
    num_rewards=30,
    grab_range=3,
)

# run the simulation: this places the rewards on the grid
follower_sim()

# add the followers to the simulation and run simulation forward
follower_sim = add_follower_foragers(
    follower_sim,
    num_follower_foragers=3,
    visibility_range=45,
    getting_worse=.5,
    optimal=3,
    proximity_decay=2,
    initial_positions=np.array([[10, 10], [20, 20], [40, 40]]),
)