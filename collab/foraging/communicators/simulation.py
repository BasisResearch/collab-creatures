import numpy as np
import pandas as pd

from .agents import Communicators
from .com_utils import loc1Dto2D


class SimulateCommunicators(object):
    def __init__(self, env, N_frames, N_agents=3, c_trust=0.5, sight_radius=5):
        self.env = env
        self.N_states = env.edge_size**2
        self.N_agents = N_agents
        self.N_frames = N_frames

        # arrays for storing simulation data
        self.food_trajectory = np.zeros([self.N_states, N_frames])

        # agents
        self.list_agents = []
        self.loc_1D_allagents = np.zeros(
            N_agents, dtype=int
        )  # list of locations occupied by an agent
        self.phi_agents = np.zeros(
            [self.N_states, 1]
        )  # how many agents in each location?

        self.all_foragersDF = pd.DataFrame()
        self.all_rewardsDF = pd.DataFrame()

        # calorie counting, not actually used in current analyses.
        self.calories_acquired_mat = np.zeros([N_agents, N_frames])
        self.calories_expended_mat = np.zeros([N_agents, N_frames])
        self.calories_total_mat = np.zeros([N_agents, N_frames])
        self.calories_cumulative_vec = np.zeros([N_agents, N_frames])

        self.calories_acquired_per_unit_time = (
            5  # TO DO: make this a property of the agent, putting it here for now
        )

        self.add_agents(
            c_trust, sight_radius
        )  # TO DO: this can be made more general and robust

        return

    def add_agents(self, c_trust, sight_radius):
        for ai in range(self.N_agents):
            new_agent = Communicators(self.env, self.N_frames, c_trust, sight_radius)
            self.list_agents.append(new_agent)

            # assign the agent a random location
            current_loc_1D = np.random.randint(self.N_states)
            new_agent.state_trajectory[0] = current_loc_1D

            # update which locations are occupied by agents
            self.loc_1D_allagents[ai] = current_loc_1D  # list
            self.phi_agents[current_loc_1D] += 1  # add an agent to this location

            self.calories_total_mat[:, 0] = 50  # each agent starts with 50 calories
        return

    def run(self):
        """Run the simulation forward num_frames time steps
        RETURN data frames


        """

        # Quantities to track
        x_agents_all = np.zeros([self.N_agents, self.N_frames])
        y_agents_all = np.zeros([self.N_agents, self.N_frames])
        # agent_locs_1d_all = np.zeros([self.N_agents, self.N_frames]) #TODO this has been never used, check!
        dist_to_nearest_neighbor_all = np.zeros([self.N_agents, self.N_frames])
        calories_acquired_all = np.zeros([self.N_agents, self.N_frames])
        # time_to_first_food_all = np.zeros([self.N_agents, 1])   #TODO this has been never used, check!

        for ti in range(self.N_frames - 1):
            # ---- Update environment --------
            # vector of how much food agents can eat at each location
            delta_food_cal = self.calories_acquired_per_unit_time * self.phi_agents

            # old version, bug, food never depletes fully
            # # rectify the calorie count for the food locations that will hit negative calories
            # is_overdepleted =  delta_food_cal > self.env.food_calories_by_loc
            # # find locations where the calorie count will hit negative values (we'll set the calorie count to 0)
            # delta_food_cal[is_overdepleted] = self.env.food_calories_by_loc[is_overdepleted]

            # eat the food
            self.env.food_calories_by_loc -= delta_food_cal
            self.env.food_calories_by_loc[self.env.food_calories_by_loc < 0] = (
                0  # set negative values to 0
            )
            self.env.phi_food = self.env.food_calories_by_loc

            # if phi_food is low, generate new food
            if np.sum(self.env.food_calories_by_loc) <= 2:
                self.env.add_food_patches()

            # if food_statistics_type == "regular_intervals":
            #     # randomly add a new food patch every several time steps
            #     if ti % epoch_dur == 0:
            #         list_newfood_loc_id = np.random.permutation(np.arange(N_states))[:N_patches]
            #         phi_food[list_newfood_loc_id] = 1
            #         food_calories_by_loc[list_newfood_loc_id] = 20
            #  add a fixed number of calories to each new food location

            self.food_trajectory[:, ti + 1] = self.env.food_calories_by_loc.flatten()

            # ---------------------Update agents ---------------------------------

            for ai, agent in enumerate(self.list_agents):
                # sum_weighted_features = agent.c.T @ features
                prev_loc_1d = int(
                    agent.state_trajectory[ti]
                )  # agent's current location

                # ------ update energy consequences of previous time step's actions --------

                # update agent's total energy based on amount of food at previous location
                # transfer calories from food to agent
                self.calories_acquired_mat[ai, ti] = (
                    delta_food_cal[prev_loc_1d] / self.phi_agents[prev_loc_1d][0]
                )[0]
                # RU: added [0] to ensure its a scalar; otherwise deprecation warning
                # RU: make sure that this is in line with your intentions
                # ELM: not currently using agent calorie counts, consider removing these variables.
                # Or, check that they are correct.

                #   # if there were N agents at that location, it gets 1/N portion of the calories
                agent.energy_total += self.calories_acquired_mat[ai, ti]
                self.calories_cumulative_vec[ai, ti + 1] = (
                    self.calories_cumulative_vec[ai, ti]
                    + self.calories_acquired_mat[ai, ti]
                )  # only tracks calories acquired?

                # -------------- Compute expected rewards, values, and make a decision --------------------------------

                self.phi_agents[prev_loc_1d] -= 1  # move out of previous location

                # EXPECTED REWARD RELATED TO OTHER AGENTS
                xloc_allagents, yloc_allagents = loc1Dto2D(
                    self.loc_1D_allagents, self.env.edge_size
                )
                xloc_self, yloc_self = loc1Dto2D(prev_loc_1d, self.env.edge_size)

                # EXPECTED REWARD RELATED TO CENTER OF MASS
                xloc_otheragents = np.delete(
                    xloc_allagents, ai
                )  # remove this agent's own location from the list
                yloc_otheragents = np.delete(yloc_allagents, ai)  #

                # VISIBILITY CONSTRAINTS
                phi_visible_mat = agent.compute_visible_locations(
                    xloc_self, yloc_self, self.env.edge_size
                )
                phi_visible = np.reshape(phi_visible_mat, (self.N_states, 1))

                # EXPECTED FOOD REWARD AT EACH LOCATION
                # information from self
                w_food_self = self.env.food_calories_by_loc * phi_visible
                # information from other foragers
                w_food_others = self.env.food_calories_by_loc * self.phi_agents

                # VALUE
                value = agent.value_update(w_food_self, w_food_others)

                # POLICY: select next action using the value and eligible states
                next_loc_1d = agent.policy_update(prev_loc_1d, value)

                # ------ Locations of agents and rewards at each time point --------
                xloc_prev, yloc_prev = loc1Dto2D(prev_loc_1d, self.env.edge_size)
                xloc_next, yloc_next = loc1Dto2D(next_loc_1d, self.env.edge_size)

                x_agents_all[ai, ti] = xloc_prev
                y_agents_all[ai, ti] = yloc_prev

                # ------- compute energy cost of moving to new location --------------
                dist_traveled = np.sqrt(
                    (xloc_next - xloc_prev) ** 2 + (yloc_next - yloc_prev) ** 2
                )
                self.calories_expended_mat[ai, ti] = (
                    agent.caloric_cost_per_unit_dist * dist_traveled
                )
                agent.energy_total -= self.calories_expended_mat[ai, ti]

                # ------------- compute metrics for data analysis -----------------
                if len(self.list_agents) > 1:
                    dist_to_neighbors = np.sqrt(
                        (xloc_otheragents - xloc_self) ** 2
                        + (yloc_otheragents - yloc_self) ** 2
                    )
                    dist_to_nearest_neighbor_all[ai, ti] = np.min(dist_to_neighbors)

                calories_acquired_all[ai, ti] = self.calories_acquired_mat[ai, ti]

                # if self.env.food_calories_by_loc[next_loc_1d][0]:
                #     agent.times_at_food.append(
                #         ti + 1
                #     )  # add this frame to the list of frames where agent is at a food location

                # -------------------------------------------------------------------

                agent.state_trajectory[ti + 1] = next_loc_1d  # scalar
                agent.value_trajectory[:, ti + 1] = (
                    value.flatten()
                )  # (N_states, N_timesteps)
                agent.energy_trajectory[ti + 1] = agent.energy_total
                self.calories_total_mat[ai, ti + 1] = (
                    self.calories_total_mat[ai, ti]
                    + self.calories_acquired_mat[ai, ti]
                    - self.calories_expended_mat[ai, ti]
                )

                self.phi_agents[next_loc_1d] += 1  # move into new location
                self.loc_1D_allagents[ai] = next_loc_1d

        # -------Save locations of each agent and reward in data frames-----------
        # all_foragersDF:
        # all_rewardsDF:

        # forager locations
        foragers_all = []
        for ai in range(self.N_agents):
            single_agent = pd.DataFrame(
                {
                    "x": x_agents_all[ai, :],
                    "y": y_agents_all[ai, :],
                    "time": range(1, self.N_frames + 1),
                    "forager": ai + 1,
                    "type": "communicators",
                }
            )

            foragers_all.append(single_agent)

        self.all_foragersDF = pd.concat(foragers_all)

        # all_foragersDF.head()

        # Reward locations
        rewards_all = []
        for ti in range(self.N_frames):
            loc1D = [
                idx for idx, val in enumerate(self.food_trajectory[:, ti]) if val > 0
            ]
            if len(loc1D) > 0:
                x_reward, y_reward = loc1Dto2D(loc1D, self.env.edge_size)
                single_time = pd.DataFrame(
                    {
                        "x": x_reward,
                        "y": y_reward,
                        "time": np.repeat(ti, len(x_reward)),
                    }
                )
                rewards_all.append(single_time)

        self.all_rewardsDF = pd.concat(rewards_all)
        return
