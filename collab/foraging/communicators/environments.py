import numpy as np

from .com_utils import create_2Dgrid, loc2Dto1D


class Environment(object):
    def __init__(
        self, edge_size=30, N_total_food_units=16, patch_dim=1, max_step_size=3
    ):
        self.edge_size = edge_size
        self.N_states = edge_size**2
        self.N_total_food_units = N_total_food_units
        self.patch_dim = patch_dim
        self.food_decay_rate = 0.3

        self.max_step_size = max_step_size

        self.x_arr, self.y_arr, self.locs_1d_arr = create_2Dgrid(edge_size)
        self.T_prob, self.T_eligible = self.build_transition_matrix(
            max_step_size=self.max_step_size
        )

        self.phi_food = np.zeros(
            [self.N_states, 1]
        )  # indicator vector showing which locations are occupied by food.
        self.phi_food_init = np.zeros([self.N_states, 1])
        self.food_calories_by_loc = np.zeros(
            [self.N_states, 1]
        )  # amount of food at each location in units of calories

        return

    def build_transition_matrix(self, max_step_size=1):
        # This function is called when the environment is initialized and the transition matrix becomes fixed.
        # The maximum step size is passed in as a parameter in the initialization function.
        # After initialization you can still call this function and pass in an arbitrary value for the max step size to
        # visualize what different transition matrices look like,
        # but it won't change the transition matrix of the environment.

        # compute eligible state transitions with a Euclidean distance rule
        # (up, down, left ,right)

        T = np.zeros([self.N_states, self.N_states])

        for i in range(self.N_states):
            for j in range(self.N_states):
                T[i, j] = (
                    np.sqrt(
                        (self.x_arr[j] - self.x_arr[i]) ** 2
                        + (self.y_arr[j] - self.y_arr[i]) ** 2
                    )
                ) <= max_step_size  # make this bigger to include more eligible states!!!

        T_eligible = T  # save the binary representation
        T_prob = T / np.sum(T, axis=0, keepdims=True)

        return T_prob, T_eligible

    # def set_transition_matrix(self, T_prob, T_eligible):
    #     self.T_prob = T_prob
    #     self.T_eligible = T_eligible
    #     return

    def add_food_patches(self):  # , food_statistics_type="drop_food_once"):
        # returns the x and y locations of the new food locations
        N_units_per_patch = self.patch_dim**2
        N_patches = np.ceil(self.N_total_food_units / N_units_per_patch).astype(int)

        # if food_statistics_type == "drop_food_once":

        for pi in range(N_patches):
            x_start = np.random.randint(0, self.edge_size - self.patch_dim)
            y_start = np.random.randint(0, self.edge_size - self.patch_dim)
            # generate (x,y) coordinates for each food unit in the patch
            x_range, y_range = np.arange(x_start, x_start + self.patch_dim), np.arange(
                y_start, y_start + self.patch_dim
            )
            x_locs, y_locs = np.meshgrid(x_range, y_range, indexing="xy")
            # convert to 1D locations
            list_newfood_loc_1d = loc2Dto1D(
                x_locs.flatten(), y_locs.flatten(), self.edge_size
            )

            # update food tracking variables
            self.phi_food[list_newfood_loc_1d] = 1
            self.phi_food_init[list_newfood_loc_1d] = 1
            self.food_calories_by_loc[
                list_newfood_loc_1d
            ] = 20  # add a fixed number of calories to each new food location

        return
