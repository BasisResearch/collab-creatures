import warnings

import numpy as np
import pandas as pd

from collab2.foraging.random_hungry_followers.rhf_helpers import (
    generate_grid,
    update_rewards,
)


class Foragers:
    def __init__(
        self,
        grid_size=30,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_foragers=3,
        num_frames=10,
        num_rewards=8,
        grab_range=2,
        include_random_foragers=False,
    ):
        """
            A class representing a simulation of random forager movements (as placeholders)
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

            num_foragers (int): The number of (randomly moving) foragers.

            num_frames (int): The number of frames in the simulation.

            num_rewards (int): The number of rewards initially
                            in the environment. Will disappear as foragers
                            grab them by being within the `grab_range`.

            grab_range (int): The range within which rewards can be grabbed.

            include_random_foragers (bool): Whether to include random foragers in
                            the final output (if you only use the sim as
                            a starting point for another foragint strategy,
                            you might want to set this to False).

        Other attributes:
            step_size_max (int): The maximum step size for forager movements
                (determined by the length of `probabilities`)
            steps (ndarray): An array of integers representing the possible
                step sizes for forager movements.
            grid (ndarray): A 2D array representing the grid-based environment.

            foragers (ndarray): An array of DataFrames representing the foragers'
                positions. Each DataFramecorresponds to a forager, each row to a
                time frame, (x, y) are the cooridnates of a given forager
                at that time.

            foragersDF (ndarray): A DataFrame representing the same information
                in a single DataFrame.

            rewards (ndarray): An a list of DataFrames representing
                the rewards' positions across frames. Each DataFrame corresponds
                to a time frame, each row to a reward.

            rewardsDF (ndarray): A DataFrame representing the same
                information in a single DataFrame.


        Methods:
            __call__(): Executes the random forager movement and rewards generation.
            generate_random_foragers(): Generates random forager movements for the simulation.
            generate_random_rewards(): Generates random rewards in the environment.

        Remarks:
            - The generated forager movements and rewards are stored in the class
              attributes `foragersDF` and `rewardsDF` respectively.
            - The simulation assumes a grid-based environment with foragers and rewards
              represented as coordinates (x, y) within the grid.
            - The forager movements are generated using the given probabilities for
              each step size, and their coordinates are bounded within the grid.
            - The rewards are randomly distributed within the grid and have a time
              value of 1 (indicating they are available from the beginning).
            - The class allows for easy simulation and analysis of forager behaviors
              and reward distributions in the specified environment.
        """

        self.grid_size = grid_size
        self.step_size_max = 4
        if len(probabilities) % 2 == 0:
            raise ValueError("The length of 'probabilities' must be odd.")
        self.probabilities = np.array(probabilities) / sum(probabilities)
        self.num_foragers = num_foragers
        self.num_frames = num_frames
        self.num_rewards = num_rewards
        self.grab_range = grab_range
        self.steps = np.arange(-self.step_size_max, self.step_size_max + 1)
        self.include_random_foragers = include_random_foragers

        self.grid = generate_grid(self.grid_size)

    def __call__(self):
        rb = self.generate_random_foragers(self.num_foragers)
        self.random_foragers = rb["random_foragers"]
        self.random_foragersDF = rb["random_foragersDF"]
        rr = self.generate_random_rewards()
        self.rewards = rr["rewards"]
        self.rewardsDF = rr["rewardsDF"]

        self.foragers = []

        if self.include_random_foragers:
            self.foragers.extend(self.random_foragers)
            self.foragersDF = pd.concat(self.foragers)

        if self.foragers:
            rew = update_rewards(self, self.rewards, self.foragers, start=1)
            self.rewards = rew["rewards"]
            self.rewardsDF = rew["rewardsDF"]

    def generate_random_foragers(self, num_foragers, size=None):
        if size is None:
            size = self.num_frames
        random_foragers = []

        size_warning_flag = False

        for forager in range(num_foragers):
            forager_x = np.cumsum(
                np.random.choice(
                    self.steps,
                    size=size,
                    p=self.probabilities,
                    replace=True,
                )
            ) + (
                self.grid_size / 2
            )  # make centered

            if any(forager_x < 0) or any(forager_x > self.grid_size):
                size_warning_flag = True
                forager_x[forager_x < 0] = 0
                forager_x[forager_x > self.grid_size] = self.grid_size

            forager_y = np.cumsum(
                np.random.choice(
                    self.steps,
                    size=size,
                    p=self.probabilities,
                    replace=True,
                )
            ) + (self.grid_size / 2)

            if any(forager_y < 0) or any(forager_y > self.grid_size):
                forager_y[forager_y < 0] = 0
                forager_y[forager_y > self.grid_size] = self.grid_size

            if size_warning_flag:
                warnings.warn(
                    "Warning: forager movements truncated to grid size. "
                    "Try running again, or increase grid size.",
                    UserWarning,
                )

            forager = pd.DataFrame(
                {
                    "x": forager_x,
                    "y": forager_y,
                    "time": range(0, size),
                    "forager": forager,
                    "type": "random",
                }
            )
            random_foragers.append(forager)

        random_foragers_data = pd.concat(random_foragers)

        return {
            "random_foragers": random_foragers,
            "random_foragersDF": random_foragers_data,
        }

    def generate_random_rewards(self, size=None):
        if size is None:
            size = self.num_rewards
        rewardsX = np.random.choice(range(1, self.grid_size + 1), size=size)
        rewardsY = np.random.choice(range(1, self.grid_size + 1), size=size)

        rewards = []
        for t in range(1, self.num_frames + 1):
            rewards.append(pd.DataFrame({"x": rewardsX, "y": rewardsY, "time": t}))

        return {"rewards": rewards, "rewardsDF": pd.concat(rewards)}


class RandomForagers(Foragers):
    def __init__(
        self,
        grid_size=30,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_foragers=3,
        num_frames=10,
        num_rewards=8,
        grab_range=2,
        include_random_foragers=True,
    ):
        super().__init__(
            grid_size=grid_size,
            probabilities=probabilities,
            num_foragers=num_foragers,
            num_frames=num_frames,
            num_rewards=num_rewards,
            grab_range=grab_range,
            include_random_foragers=include_random_foragers,
        )
