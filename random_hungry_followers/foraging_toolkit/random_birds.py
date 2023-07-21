import sys

sys.path.insert(0, "..")

# print(sys.path)
import foraging_toolkit as ft

import pandas as pd
import numpy as np

# import foraging_toolkit as ft

from itertools import product

import pandas as pd
import numpy as np


class RandomBirds:
    def __init__(
        self,
        grid_size=30,
        step_size_max=4,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_birds=3,
        num_frames=10,
        num_rewards=8,
        grab_range=2,
    ):
        self.grid_size = grid_size
        self.step_size_max = step_size_max
        self.probabilities = np.array(probabilities) / sum(probabilities)
        self.num_birds = num_birds
        self.num_frames = num_frames
        self.num_rewards = num_rewards
        self.grab_range = grab_range
        self.steps = np.arange(-step_size_max, step_size_max + 1)

        self.grid = ft.generate_grid(self.grid_size)

    def __call__(self):
        self.generate_random_birds()
        self.generate_random_rewards()
        rw = ft.update_rewards(
            self.rewards,
            self.birds,
            self.num_birds,
            self.num_frames,
            self.grab_range,
        )
        self.rewards = rw["rewards"]
        self.rewardsDF = rw["rewardsDF"]

    def generate_random_birds(self):  # generate birds
        self.birds = []

        for bird in range(self.num_birds):
            bird_x = np.cumsum(
                np.random.choice(
                    self.steps,
                    size=self.num_frames,
                    p=self.probabilities,
                    replace=True,
                )
            ) + (
                self.grid_size / 2
            )  # make centered

            bird_x[bird_x < 0] = 0
            bird_x[bird_x > self.grid_size] = self.grid_size

            bird_y = np.cumsum(
                np.random.choice(
                    self.steps,
                    size=self.num_frames,
                    p=self.probabilities,
                    replace=True,
                )
            ) + (self.grid_size / 2)

            bird_y[bird_y < 0] = 0
            bird_y[bird_y > self.grid_size] = self.grid_size

            bird = pd.DataFrame({"x": bird_x, "y": bird_y})
            self.birds.append(bird)

        bird_data = pd.concat(self.birds)
        bird_data["bird"] = pd.Categorical(
            np.repeat(range(1, self.num_birds + 1), self.num_frames)
        )
        bird_data["time"] = np.tile(
            range(1, self.num_frames + 1), self.num_birds
        )

        # remove later, as it is very likely redundant
        bird_data.loc[bird_data["x"] < 0, "x"] = 0
        bird_data.loc[bird_data["x"] > self.grid_size, "x"] = self.grid_size
        bird_data.loc[bird_data["y"] < 0, "y"] = 0
        bird_data.loc[bird_data["y"] > self.grid_size, "y"] = self.grid_size

        self.birdsDF = bird_data

    def generate_random_rewards(self):
        rewardsX = np.random.choice(
            range(1, self.grid_size + 1), size=self.num_rewards
        )
        rewardsY = np.random.choice(
            range(1, self.grid_size + 1), size=self.num_rewards
        )

        rewards = []
        rewards.append(pd.DataFrame({"x": rewardsX, "y": rewardsY, "time": 1}))

        self.rewards = rewards
        self.rewardsDF = pd.concat(self.rewards)
