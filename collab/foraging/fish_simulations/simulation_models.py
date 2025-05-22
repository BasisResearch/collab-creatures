import numpy as np
from typing import Any, Optional


class FishSchoolSimulation:
    """
    Simulate fish schooling using Vicsek alignment + cohesion/repulsion forces
    in a circular tank with reflective boundary conditions.
    """
    
    def __init__(
        self, 
        N: Optional[int] = 4,
        arena_size: Optional[float] = 50,
        Tmax: Optional[float] = 10,
        dt: Optional[float] = 0.1,
        random_state: Optional[int] = 0,
        interaction_params: Optional[dict[str, dict[str, Any]]] = None,
        v_scale : Optional[float] = 5,

    ):
        """
        :param N: number of fish
        :param arena_size: Radius of arena
        :param Tmax: simulation end time
        :param dt: time-step for logging fish positions, speed, and heading. 
        :param random_state: random seed for reproducibility
        :param interaction_params: nested dictionary of parameters of the interactions in the model.
            the keys of the dictionary are interaction names, each corresponding value is a dictionary of
            parameters for the interaction
        :param v_scale: Scale used to sample initial velocity magnitude from Rayleigh distribution
        """
        self.N = N
        self.arena_size = arena_size
        self.Tmax = Tmax
        self.dt = dt 
        self.interaction_params = interaction_params
        self.v_scale = v_scale 

        # initialize random number generator
        self.rng = np.random.default_rng(seed=random_state)

        # choose default params if interaction_params not specified
        if self.interaction_params is None :
            self.interaction_params = {
                "vicsek": {
                    "weight": 1 * np.ones(self.N),
                    "interaction_length": self.arena_size / 3 * np.ones(self.N),
                    "sigma_v": v_scale/2 * np.ones(self.N),
                    "sigma_t": np.pi/4 * np.ones(self.N), 
                },
                "cohesion": {
                    "weight": 1 * np.ones(self.N),
                    "interaction_length": self.arena_size / 3 * np.ones(self.N),
                    "r_0": 5 * np.ones(self.N),
                    "k_attraction": 1 * np.ones(self.N),
                    "k_repulsion": 5 * np.ones(self.N),
                }
            }
        
        self.trajectories = None
        self.velocities = None

    
    def initialize_fish(self):
        """Initialize fish positions, velocity and orientations.""" 

        self.time = 0        
        # initialize array sizes
        # add 1 since we also log time=0
        total_time_steps = np.ceil(self.Tmax/self.dt).astype(int) + 1
        self.total_time_steps = total_time_steps

        # trajectories matrix has shape (N x total_time_steps x 2), with the last dimension representing x,y positions
        self.trajectories = np.full((self.N, total_time_steps, 2), np.nan)
        
        # velocities matrix has shape (N x total_time_steps x 2), with the last column representing v, theta
        self.velocities = np.full((self.N, total_time_steps, 2), np.nan)

        # randomly choose intial conditions
        # choose random positions inside the arena from uniform distribution
        r1 = self.arena_size * self.rng.random(self.N)
        r2 = 2 * np.pi * self.rng.random(self.N)
        self.trajectories[:, 0, 0] = r1 * np.cos(r2)
        self.trajectories[:, 0, 1] = r1 * np.sin(r2)

        # choose theta from uniform distribution, v from rayleigh
        self.velocities[:, 0, 1] = (
            2 * np.pi * self.rng.random(self.N) - np.pi
        )  # heading angles in [-np.pi, np.pi)

        self.velocities[:, 0 ,0] = self.rng.rayleigh(scale=self.vscale,size=self.N)


    def dtheta_vicsek(self, f, t):
        """
        Calculate change in heading due to vicsek alignment for fish f at time t
        :param f: fish index
        :param t: time index
        :return: change in heading
        """

        # get parameters
        weight = self.interaction_params["vicsek"]["weight"][f]
        interaction_length = self.interaction_params["vicsek"]["interaction_length"][f]
        sigma_t = self.interaction_params["vicsek"]["sigma_t"][f]

        # get neighbors within interaction length
        distance = np.linalg.norm(self.trajectories[:, t, :] - self.trajectories[f, t, :], axis=1)
        neighbors = np.where((distance < interaction_length) & (distance > 0))[0]

        if len(neighbors) == 0:
            return 0
        
        # calculate average heading of neighbors
        u_x = np.mean(np.cos(self.velocities[neighbors, t, 1]))
        u_y = np.mean(np.sin(self.velocities[neighbors, t, 1]))

        if u_x == 0 and u_y == 0:
            return 0
        
        preferred_theta = np.arctan2(u_y, u_x) + sigma_t * self.rng.normal()
        # ensure preferred_theta is in [-pi, pi)
        preferred_theta = np.arctan2(np.sin(preferred_theta), np.cos(preferred_theta))

        return weight * (preferred_theta - self.velocities[f, t, 1])

    def dtheta_cohesion(self, f, t):
        """
        Calculate change in heading due to cohesion/repulsion for fish f at time t
        :param f: fish index
        :param t: time index
        :return: change in heading
        """

        # get parameters
        weight = self.interaction_params["cohesion"]["weight"][f]
        interaction_length = self.interaction_params["cohesion"]["interaction_length"][f]
        r_0 = self.interaction_params["cohesion"]["r_0"][f]
        k_attraction = self.interaction_params["cohesion"]["k_attraction"][f]
        k_repulsion = self.interaction_params["cohesion"]["k_repulsion"][f]

        # get neighbors within interaction length
        distance = np.linalg.norm(self.trajectories[:, t, :] - self.trajectories[f, t, :], axis=1)
        neighbors = np.where((distance < interaction_length) & (distance > 0))[0]

        if len(neighbors) == 0:
            return 0
        
        total_force = np.zeros(2)

        for n in neighbors:
            u_ij = (self.trajectories[n, t, :] - self.trajectories[f, t, :])/distance[n]
            if distance[n] < r_0:
                total_force += k_repulsion * (distance[n] - r_0) * u_ij
            else:
                total_force += k_attraction * (distance[n] - r_0) * u_ij
        
        # calculate angle of total force
        force_angle = np.arctan2(total_force[1], total_force[0])

        return weight * (force_angle - self.velocities[f, t, 1])
    
    def reflective_boundary(self, t):
        """
        Implement reflective boundary conditions for position/velocity at time t
        """

        x = self.trajectories[:, t, 0]
        y = self.trajectories[:, t, 1]
        theta = self.velocities[:, t, 1]

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
                self.velocities[i, t, 1] = np.arctan2(hy_new, hx_new)
                # Note that velocity magnitude is unchanged by reflection!

                # Set the new position to the intersection point with the boundary
                self.trajectories[i, t, 0] = self.arena_size * ux
                self.trajectories[i, t, 1] = self.arena_size * uy

    def update_positions(self, t):
        """
        Evolve kinematic variables at t to get position/velocity at time t+1
        """
        # update positions
        vx = self.velocities[:, t, 0] * np.cos(self.velocities[:, t, 1])
        vy = self.velocities[:, t, 0] * np.sin(self.velocities[:, t, 1])
        self.trajectories[:, t+1, 0] = (
            self.trajectories[:, t, 0] + vx * self.dt
        )
        self.trajectories[:, t+1, 1] = (
            self.trajectories[:, t, 1] + vy * self.dt
        )

        # copy previous velocities
        self.velocities[:, t+1, :] = self.velocities[:, t, :]

        # apply boundary conditions
        self.reflective_boundary(t+1)

    def step(self):
        """
        Evolve the simulation by one time step
        """
        self.update_positions(t = self.time)
        self.time += 1

        # get new heading angles and velocity 
        for f in range(self.N):
            dtheta = (
                self.dtheta_vicsek(f, self.time) + self.dtheta_cohesion(f, self.time)
            )
            self.velocities[f, self.time, 1] += dtheta
            
            # add Gaussian noise to speed
            self.velocities[f, self.time, 0] += (
                self.interaction_params["vicsek"]["sigma_v"][f]
                * self.rng.normal()
            )

        # ensure heading is in [-pi, pi)
        self.velocities[:, self.time, 1] = np.arctan2(
            np.sin(self.velocities[:, self.time, 1]),
            np.cos(self.velocities[:, self.time, 1]),
        )
        # clipping speed to be non-negative
        self.velocities[:, self.time, 0] = np.clip(
            self.velocities[:, self.time, 0], 0, None
        )


    def run_simulation(self):
        """
        Run the simulation for Tmax time
        """
        self.initialize_fish()

        # run simulation
        while self.time < self.total_time_steps:
            self.step()





        
        