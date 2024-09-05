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
        vicsek_params: Optional[dict[str, Any]] = None,
        pairwiseCopying_params: Optional[dict[str, Any]] = None,
        diffusion_params: Optional[dict[str, Any]] = None,
    ):
        """
        :param N: number of fish
        :param arena_size: Radius of arena
        :param Tmax: simulation end time
        :param dt: time-step for logging fish positions, speed, and heading
        :param random_state: random seed for reproducibility
        :param vicsek_params: parameters of vicsek interactions in the simulations
            Must include keys: "rate", "interaction_length", "sigma_v", "sigma_t".
            Each value must be a list of length = N, indicating parameter choices for each fish
            in the simulation.
        :param pairwiseCopying_param: parameters of pairwise copying interactions in the simulations
            Must include keys: "rate", "interaction_length", "sigma_v", "sigma_t".
            Each value must be a list of length = N, indicating parameter choices for each fish
            in the simulation.
        :param diffusion_params: parameters for velocity diffusion in the simulations
            Must include keys: "rate", "sigma_v", "sigma_t".
            Each value must be a list of length = N, indicating parameter choices for each fish
            in the simulation.
        """
        self.N = N
        self.arena_size = arena_size
        self.Tmax = Tmax
        self.dt = dt

        # choose default parameters
        if vicsek_params is None:
            vicsek_params = {
                "rate": np.ones((N, 1)),
                "interaction_length": arena_size / 3 * np.ones((N, 1)),
                "sigma_v": 5 * np.ones((N, 1)),
                "sigma_t": 1 * np.ones((N, 1)),
            }

        if pairwiseCopying_params is None:
            pairwiseCopying_params = {
                "rate": np.ones((N, 1)),
                "interaction_length": arena_size / 3 * np.ones((N, 1)),
                "sigma_v": 5 * np.ones((N, 1)),
                "sigma_t": 1 * np.ones((N, 1)),
            }

        if diffusion_params is None:
            diffusion_params = {
                "rate": np.ones((N, 1)),
                "sigma_v": 5 * np.ones((N, 1)),
                "sigma_t": 1 * np.ones((N, 1)),
            }

        self.vicsek_params = vicsek_params
        self.pairwiseCopying_params = pairwiseCopying_params
        self.diffusion_params = diffusion_params

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

    def evolve_positions(self, t_ind, nSteps):
        """
        Evolve kinematic variables for all fish for `nSteps` time steps. starting at timestep t_ind
        """

        for i in range(nSteps):
            # update positions
            vx = self.velocities[:, t_ind, 0] * np.cos(self.velocities[:, t_ind, 1])
            vy = self.velocities[:, t_ind, 0] * np.sin(self.velocities[:, t_ind, 1])
            self.trajectories[:, t_ind + 1, 0] = (
                self.trajectories[:, t_ind, 0] + vx * self.dt
            )
            self.trajectories[:, t_ind + 1, 1] = (
                self.trajectories[:, t_ind, 1] + vy * self.dt
            )

            # copy previous velocities
            self.velocities[:, t_ind + 1, 0] = self.velocities[:, t_ind, 0]
            self.velocities[:, t_ind + 1, 1] = self.velocities[:, t_ind, 1]

            # apply boundary conditions
            self.apply_reflective_bc(t_ind + 1)

            # update time
            self.time += self.dt
            t_ind += 1

    def simulate(self):

        nSteps = np.ceil(self.Tmax / self.dt).astype(int)
        self.evolve_positions(t_ind=0, nSteps=nSteps)
