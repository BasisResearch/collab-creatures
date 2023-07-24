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
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_birds=3,
        num_frames=10,
        num_rewards=8,
        grab_range=2,
    ):
        """
            A class representing a simulation of random bird movements and rewards
        in a grid-based environment.

        Args (included among the attributes):
            grid_size (int): The size of the grid representing the environment.
                    If it's too small, a warning will be issued and moves
                    which otherwise would
                    end up outside of the grid will be
                    truncated to the grid's edge.

            probabilities (list): A list of (potentially) unnormalized
                probabilities for each step size in
                `[- step_size_max, + step_size_max]`.
                Will be normalized in the computation.

            num_birds (int): The number of (randomly moving) birds.

            num_frames (int): The number of frames in the simulation.

            num_rewards (int): The number of rewards initially
                            in the environment. Will disappear as birds
                            grab them by being within the `grab_range`.

            grab_range (int): The range within which rewards can be grabbed.

        Other attributes:
            step_size_max (int): The maximum step size for bird movements
                (determined by the length of `probabilities`)
            steps (ndarray): An array of integers representing the possible
                step sizes for bird movements.
            grid (ndarray): A 2D array representing the grid-based environment.

            birds (ndarray): An array of DataFrames representing the birds'
                positions. Each DataFramecorresponds to a bird, each row to a
                time frame, (x, y) are the cooridnates of a given bird
                at that time.

            birdsDF (ndarray): A DataFrame representing the same information
                in a single DataFrame.

            rewards (ndarray): An a list of DataFrames representing
                the rewards' positions across frames. Each DataFrame corresponds
                to a time frame, each row to a reward.

            rewardsDF (ndarray): A DataFrame representing the same
                information in a single DataFrame.



        Methods:
            __call__(): Executes the random bird movement and rewards generation.
            generate_random_birds(): Generates random bird movements for the simulation.
            generate_random_rewards(): Generates random rewards in the environment.

        Remarks:
            - The generated bird movements and rewards are stored in the class
              attributes `birdsDF` and `rewardsDF` respectively.
            - The simulation assumes a grid-based environment with birds and rewards
              represented as coordinates (x, y) within the grid.
            - The bird movements are generated using the given probabilities for
              each step size, and their coordinates are bounded within the grid.
            - The rewards are randomly distributed within the grid and have a time
              value of 1 (indicating they are available from the beginning).
            - The class allows for easy simulation and analysis of bird behaviors
              and reward distributions in the specified environment.
        """

        self.grid_size = grid_size
        self.step_size_max = 4
        self.probabilities = np.array(probabilities) / sum(probabilities)
        self.num_birds = num_birds
        self.num_frames = num_frames
        self.num_rewards = num_rewards
        self.grab_range = grab_range
        self.steps = np.arange(-self.step_size_max, self.step_size_max + 1)

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
