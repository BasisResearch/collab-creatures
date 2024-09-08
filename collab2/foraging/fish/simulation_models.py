from typing import Any, Optional

import numpy as np


class Fish_IndependentRates:
    def __init__(
        self,
        N: Optional[int] = 4,
        arena_size: Optional[float] = 50,
        Tmax: Optional[float] = 10,
        dt: Optional[float] = 0.1,
        random_state: Optional[int] = 0,
        interaction_params: Optional[dict[str, dict[str, Any]]] = None,
    ):
        """
        :param N: number of fish
        :param arena_size: Radius of arena
        :param Tmax: simulation end time
        :param dt: time-step for logging fish positions, speed, and heading
        :param random_state: random seed for reproducibility
        :param interaction_params: nested dictionary of parameters of the interactions in the model.
            the keys of the dictionary are interaction names, each corresponding value is a dictionary of
            parameters for the interaction, eg, "rate" (required for all interactions). There must exist a method
            named "{interaction}_interaction" to implement each interaction.
        """
        self.N = N
        self.arena_size = arena_size
        self.Tmax = Tmax
        self.dt = dt

        # choose default parameters
        if interaction_params is None:
            interaction_params = {
                "vicsek": {
                    "rate": 0.5 * np.ones(N),
                    "interaction_length": arena_size / 3 * np.ones(N),
                    "sigma_v": 5 * np.ones(N),
                    "sigma_t": 1 * np.ones(N),
                },
                "pairwiseCopying": {
                    "rate": 0.5 * np.ones(N),
                    "interaction_length": arena_size / 3 * np.ones(N),
                    "sigma_v": 5 * np.ones(N),
                    "sigma_t": 1 * np.ones(N),
                },
                "diffusion": {
                    "rate": 0.5 * np.ones(N),
                    "sigma_v": 5 * np.ones(N),
                    "sigma_t": 1 * np.ones(N),
                },
            }
        self.interaction_params = interaction_params

        ## initialize matrices to save trajectory and velocity information ##
        time_steps = np.ceil(Tmax / dt).astype(int) + 1

        # trajectories matrix has shape (N x time_steps x 2), with the last column representing x,y
        self.trajectories = np.full((N, time_steps, 2), np.nan)
        # velocities matrix has shape (N x time_steps x 2), with the last column representing v, theta
        self.velocities = np.full((N, time_steps, 2), np.nan)

        ## choose random initial conditions ##
        self.time = 0

        # initialize random number generator
        self.rng = np.random.default_rng(seed=random_state)

        # random positions inside the arena
        r1 = arena_size * self.rng.random(N)
        r2 = 2 * np.pi * self.rng.random(N)
        self.trajectories[:, 0, 0] = r1 * np.cos(r2)
        self.trajectories[:, 0, 1] = r1 * np.sin(r2)

        # random velocity magnitude and direction
        self.velocities[:, 0, 0] = arena_size / 10 * self.rng.random(N)
        self.velocities[:, 0, 1] = (
            2 * np.pi * self.rng.random(N) - np.pi
        )  # heading angles in [-np.pi, np.pi)

    def apply_reflective_bc(self, t_ind):
        """
        Implement reflective boundary conditions (w/ a circular arena of R=arena_size) for row index `t_ind` of trajectory, velocity matrices
        """

        x = self.trajectories[:, t_ind, 0]
        y = self.trajectories[:, t_ind, 1]
        theta = self.velocities[:, t_ind, 1]

        distances_from_center = np.sqrt(x**2 + y**2)

        for i, d in enumerate(distances_from_center):
            if d > self.arena_size:
                # Calculate the unit vector pointing from the center of the arena to the current position
                ux = x[i] / d
                uy = y[i] / d

                # Calculate the projection of the heading direction onto this unit vector
                h_parallel = np.cos(theta[i]) * ux + np.sin(theta[i]) * uy

                # Calculate the reflection of the heading vector
                hx_new = np.cos(theta[i]) - 2 * h_parallel * ux
                hy_new = np.sin(theta[i]) - 2 * h_parallel * uy

                # Calculate new heading angle, and update velocity matrix
                self.velocities[i, t_ind, 1] = np.arctan2(hy_new, hx_new)
                # Note that velocity magnitude is unchanged by reflection!

                # Set the new position to the intersection point with the boundary
                self.trajectories[i, t_ind, 0] = self.arena_size * ux
                self.trajectories[i, t_ind, 1] = self.arena_size * uy

    def evolve_positions(self, idx_0, idx_f):
        """
        Evolve kinematic variables for all fish from idx_0 to idx_f
        """
        while idx_0 < idx_f:
            # update positions
            vx = self.velocities[:, idx_0, 0] * np.cos(self.velocities[:, idx_0, 1])
            vy = self.velocities[:, idx_0, 0] * np.sin(self.velocities[:, idx_0, 1])
            self.trajectories[:, idx_0 + 1, 0] = (
                self.trajectories[:, idx_0, 0] + vx * self.dt
            )
            self.trajectories[:, idx_0 + 1, 1] = (
                self.trajectories[:, idx_0, 1] + vy * self.dt
            )

            # copy previous velocities
            self.velocities[:, idx_0 + 1, 0] = self.velocities[:, idx_0, 0]
            self.velocities[:, idx_0 + 1, 1] = self.velocities[:, idx_0, 1]

            # apply boundary conditions
            self.apply_reflective_bc(idx_0 + 1)

            # update index
            idx_0 += 1

    def time_to_next_interaction(self):
        total_rate = np.sum(
            [
                np.sum(self.interaction_params[key]["rate"])
                for key in self.interaction_params.keys()
            ]
        )
        r1 = self.rng.random()
        return 1 / total_rate * np.log(1 / r1)

    def sample_next_interaction(self):
        rate_matrix = np.stack(
            (
                [
                    self.interaction_params[key]["rate"]
                    for key in self.interaction_params.keys()
                ]
            ),
            axis=1,
        )  # each fish corresponds to a separate row

        p_fish = np.sum(rate_matrix, axis=1) / np.sum(rate_matrix, axis=None)

        # sample a fish
        f = self.rng.choice(range(self.N), p=p_fish)

        # sample which reaction occurs
        interactions_list = list(self.interaction_params.keys())
        p_interactions = rate_matrix[f, :] / np.sum(rate_matrix[f, :])
        interaction = self.rng.choice(interactions_list, p=p_interactions)
        return f, interaction

    def find_neighbors(self, f, t_ind, L):
        """
        find neighbors of fish f at time index t_ind given interaction length L
        """
        dist2 = (
            self.trajectories[f, t_ind, 0] - self.trajectories[:, t_ind, 0]
        ) ** 2 + (self.trajectories[f, t_ind, 1] - self.trajectories[:, t_ind, 1]) ** 2
        dist2[f] = np.nan
        neighbors = np.flatnonzero(dist2 < L**2)
        return neighbors

    def vicsek_interaction(self, f):
        t_ind = int(self.time / self.dt)
        L = self.interaction_params["vicsek"]["interaction_length"][f]
        sigma_v = self.interaction_params["vicsek"]["sigma_v"][f]
        sigma_t = self.interaction_params["vicsek"]["sigma_t"][f]

        neighbors = self.find_neighbors(f, t_ind, L)

        if len(neighbors):
            # find average velocity
            vx = np.mean(
                self.velocities[neighbors, t_ind, 0]
                * np.cos(self.velocities[neighbors, t_ind, 1])
            )
            vy = np.mean(
                self.velocities[neighbors, t_ind, 0]
                * np.sin(self.velocities[neighbors, t_ind, 1])
            )
            v = np.sqrt(vx**2 + vy**2)
            theta = np.arctan2(vy, vx)

            # add noise and update f
            self.velocities[f, t_ind, 0] = v + self.rng.normal(loc=0, scale=sigma_v)
            self.velocities[f, t_ind, 1] = theta + self.rng.normal(loc=0, scale=sigma_t)

            # make sure velocity magnitude is not negative!
            self.velocities[f, t_ind, 0] = np.maximum(0, self.velocities[f, t_ind, 0])

    def pairwiseCopying_interaction(self, f):
        t_ind = int(self.time / self.dt)
        L = self.interaction_params["pairwiseCopying"]["interaction_length"][f]
        sigma_v = self.interaction_params["pairwiseCopying"]["sigma_v"][f]
        sigma_t = self.interaction_params["pairwiseCopying"]["sigma_t"][f]

        neighbors = self.find_neighbors(f, t_ind, L)

        if len(neighbors):
            # choose a neighbor to randomly copy
            j = self.rng.choice(neighbors)

            # add noise and update f
            self.velocities[f, t_ind, 0] = self.velocities[
                j, t_ind, 0
            ] + self.rng.normal(loc=0, scale=sigma_v)
            self.velocities[f, t_ind, 1] = self.velocities[
                j, t_ind, 1
            ] + self.rng.normal(loc=0, scale=sigma_t)

            # make sure velocity magnitude is not negative!
            self.velocities[f, t_ind, 0] = np.maximum(0, self.velocities[f, t_ind, 0])

    def diffusion_interaction(self, f):
        t_ind = int(self.time / self.dt)
        sigma_v = self.interaction_params["diffusion"]["sigma_v"][f]
        sigma_t = self.interaction_params["diffusion"]["sigma_t"][f]
        # add noise
        self.velocities[f, t_ind, 0] += self.rng.normal(loc=0, scale=sigma_v)
        self.velocities[f, t_ind, 1] += self.rng.normal(loc=0, scale=sigma_t)
        # make sure velocity magnitude is not negative!
        self.velocities[f, t_ind, 0] = np.maximum(0, self.velocities[f, t_ind, 0])

    def simulate(self):
        # t_init = 0
        # nSteps = np.ceil(self.Tmax/self.dt).astype(int)
        # self.evolve_positions(t_init, nSteps)

        while True:
            # find time to next reaction
            tau = self.time_to_next_interaction()

            # evolve positions until next reaction
            idx_0 = int(self.time / self.dt)
            idx_f = np.minimum(
                idx_0 + np.round(tau / self.dt), np.ceil(self.Tmax / self.dt)
            ).astype(int)
            self.evolve_positions(idx_0, idx_f)

            # update time in increments of dt
            self.time = idx_f * self.dt

            # choose which reaction occurs
            f, interaction = self.sample_next_interaction()

            # apply transformation
            interaction_func = getattr(self, f"{interaction}_interaction")
            interaction_func(f)

            if self.time >= self.Tmax:
                break
