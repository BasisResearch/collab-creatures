import numpy as np
from .com_utils import softmax


class Communicators(object):
    def __init__(self, env, N_timesteps, c_trust=0.5, sight_radius=5, discount_factor=0.9, energy_init=50):
        self.env = env
        self.c_trust = c_trust  # how much the agent trusts information about rewards from other agents
        self.discount_factor = discount_factor  # scalar between 0 and 1
        self.SR = np.linalg.pinv(np.eye(env.N_states) - discount_factor * env.T_prob)  # (N_states, N_states) matrix
        self.state_trajectory = np.zeros([N_timesteps])
        self.value_trajectory = np.zeros([env.N_states, N_timesteps])
        self.phi_neighbors = np.zeros(
            [env.N_states, 1]
        )  # indicator  vector showing which locations do I see a neighboring agent?
        self.energy_total = energy_init  # scalar quantity (calories)
        self.energy_trajectory = np.zeros([N_timesteps])  # not sure if we need this to be inside the class
        self.energy_trajectory[0] = energy_init
        # self.times_at_food = [] # list of frames the agent is at a food location
        self.sight_radius = sight_radius
        self.caloric_cost_per_unit_dist = 1

        self.caloric_cost_per_unit_dist = 1
        self.doProbabilisticPolicy = True
        self.doSoftmaxPolicy = True
        self.exploration_bias = 0.001

        self.weights_list = [c_trust]

        return

    def compute_visible_locations(self, xloc_self, yloc_self, edge_size):
        phi_visible_mat = np.zeros([edge_size, edge_size])
        x_arr = np.arange(edge_size)
        y_arr = np.arange(edge_size)
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing="xy")
        phi_visible_mat = np.sqrt((x_grid - xloc_self) ** 2 + (y_grid - yloc_self) ** 2) <= self.sight_radius

        return phi_visible_mat

    def value_update(self, w_food_self, w_food_others):
        # w_food_others -- information from other birds about locations of food rewards
        # w_food_self -- my own information about locations of food rewards
        sum_weighted_features = (1 - self.c_trust) * w_food_self + self.c_trust * w_food_others
        value = self.SR @ sum_weighted_features  # (N_states, 1) vector
        return value

    def policy_update(self, prev_loc_1d, value):
        # eligible states are those specified by the transition matrix. Can constrain further to exclude states not occupied by other agents
        eligible_states_id = np.nonzero(self.env.T_eligible[:, prev_loc_1d])[
            0
        ]  # * np.logical_not(phi_agents.flatten()))[0]       # state IDs of eligible states
        value_eligible = value[eligible_states_id].flatten()  # value of eligible states
        # value_eligible += 0.001 * np.random.rand(value_eligible.shape[0]) # add some noise

        # ACTION SELECTION

        if self.doProbabilisticPolicy:
            if self.doSoftmaxPolicy:
                prob_arr = softmax(value_eligible, temp=self.exploration_bias)
            else:
                # #sample eligible states from a categorical distribution whose shape is based on the values
                # # convert values into probabilities
                # value_eligible += 0.001 * np.random.randn(value_eligible.shape[0]) # add some noise
                # prob_arr = value_eligible - np.min(value_eligible) # shift values so they are all positive and add some noise
                prob_arr = value_eligible - np.min(value_eligible)
                prob_arr += np.mean(value_eligible) * 0.001 * np.random.rand(value_eligible.shape[0])
                prob_arr = prob_arr / np.sum(prob_arr)  # normalize so they sum to 1

            next_loc_1d = np.random.choice(eligible_states_id, p=prob_arr)

        else:
            # DETERMINISTIC POLICY
            next_loc_1d = eligible_states_id[np.argmax(value_eligible)]
        return next_loc_1d
