import sys

sys.path.insert(0, "..")

import foraging_toolkit as ft

import pandas as pd
import numpy as np
import warnings

from itertools import product

import pandas as pd
import numpy as np


class Birds:
    def __init__(
        self,
        grid_size=30,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_birds=3,
        num_frames=10,
        num_rewards=8,
        grab_range=2,
        include_random_birds=False,
    ):
        """
            A class representing a simulation of random bird movements (as placeholders)
            and rewards in a grid-based environment.

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

            include_random_birds (bool): Whether to include random birds in
                            the final output (if you only use the sim as
                            a starting point for another foragint strategy,
                            you might want to set this to False).

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
        self.include_random_birds = include_random_birds

        self.grid = ft.generate_grid(self.grid_size)

    def __call__(self):
        self.generate_random_birds()
        self.generate_random_rewards()

        self.birds = []

        if self.include_random_birds:
            self.birds.extend(self.random_birds)
            self.birdsDF = pd.concat(self.birds)

        if self.birds:
            ft.update_rewards(self)

    def generate_random_birds(self):
        self.random_birds = []

        size_warning_flag = False

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

            if any(bird_x < 0) or any(bird_x > self.grid_size):
                size_warning_flag = True
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

            if any(bird_y < 0) or any(bird_y > self.grid_size):
                bird_y[bird_y < 0] = 0
                bird_y[bird_y > self.grid_size] = self.grid_size

            if size_warning_flag:
                warnings.warn(
                    "Warning: bird movements truncated to grid size. "
                    "Increase grid size to avoid this.",
                    UserWarning,
                )

            bird = pd.DataFrame(
                {
                    "x": bird_x,
                    "y": bird_y,
                    "time": range(1, self.num_frames + 1),
                    "bird": bird + 1,
                    "type": "random",
                }
            )
            self.random_birds.append(bird)

        random_bird_data = pd.concat(self.random_birds)

        self.random_birdsDF = random_bird_data

    def generate_random_rewards(self):
        rewardsX = np.random.choice(range(1, self.grid_size + 1), size=self.num_rewards)
        rewardsY = np.random.choice(range(1, self.grid_size + 1), size=self.num_rewards)

        rewards = []
        rewards.append(pd.DataFrame({"x": rewardsX, "y": rewardsY, "time": 1}))

        self.rewards = rewards
        self.rewardsDF = pd.concat(self.rewards)


class RandomBirds(Birds):
    def __init__(
        self,
        grid_size=30,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_birds=3,
        num_frames=10,
        num_rewards=8,
        grab_range=2,
        include_random_birds=True,
    ):
        super().__init__(
            grid_size=grid_size,
            probabilities=probabilities,
            num_birds=num_birds,
            num_frames=num_frames,
            num_rewards=num_rewards,
            grab_range=grab_range,
            include_random_birds=include_random_birds,
        )
