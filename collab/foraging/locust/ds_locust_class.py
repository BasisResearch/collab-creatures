import os

import dill
import torch

import pyro
from pyro.infer import Predictive

import matplotlib.pyplot as plt

from .ds_locust_functions import (
    get_count_data_subset,
    LocustDynamics,
    conditioned_locust_model,
    simulated_bayesian_locust,
    plot_ds_estimates
)

from collab.utils import find_repo_root
from collab.foraging.toolkit.dynamical_utils import (
plot_ds_trajectories,
run_svi_inference,
ds_uncertainty_plot,
)

from chirho.dynamical.handlers import LogTrajectory, StaticBatchObservation
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import Dynamics, State, simulate

class LocustDS():

    def __init__(self, data_code, start, end):
        
        self.data_code = data_code
        self.start = start
        self.start_tensor = torch.tensor(start).float()
        self.end = end
        self.end_tensor = torch.tensor(end).float()

        self.root = find_repo_root()
        self.data_path =  os.path.join(
            self.root, f"data/foraging/locust/ds/locust_counts{data_code}.pkl"
        )

        with open(self.data_path, "rb") as f:
            locust_count_data = dill.load(f)

            self.count_data = locust_count_data["count_data"]
        
        self.logging_times = torch.arange(start, end, 1)
        self.c_data = get_count_data_subset(self.count_data, 
                                                     self.start, 
                                                     self.end)
        self.subset = self.c_data["count_subset"]
        self.init_state = self.c_data["init_state"]

        self.piecemeal_path = os.path.join(self.root, 
                                      "data/foraging/locust/ds/")

    def simulate_trajectories(self, true_attraction,
                            true_wander, init_state = None):
        
        if init_state is None:
            init_state = self.init_state
        
        locust_true = LocustDynamics(true_attraction, true_wander)
        with TorchDiffEq(method="rk4", ), LogTrajectory(self.logging_times) as lt:
            simulate(locust_true, init_state, self.start_tensor, self.end_tensor)

        self.simulated_traj = lt.trajectory

    def plot_simulated_trajectories(self,window_size=0):
        plot_ds_trajectories(self.simulated_traj, self.logging_times, 
                                                   window_size=window_size)
        
    def get_prior_samples(self, num_samples, force = False):
        
        self.priors_path = os.path.join(self.piecemeal_path, 
            f"priors_sam{num_samples}.pkl")
        if os.path.exists(self.priors_path) and not force:
            with open(self.priors_path, 'rb') as file:
                self.prior_samples = dill.load(file)
        
        else:
            prior_predictive = Predictive(simulated_bayesian_locust, num_samples=num_samples)
            prior_samples = prior_predictive(self.init_state, 
                                                self.start_tensor,
                                                self.logging_times)
            self.prior_samples =  prior_samples

            with open(self.priors_path, 'wb') as file:
                dill.dump(prior_samples, file)

    def run_inference(self, name, num_iterations, lr = .003, num_samples = 50, force = False):
        
        self.file_path = os.path.join(self.piecemeal_path, 
            f"{name}_s{self.start}_e{self.end}_i{num_iterations}_{self.data_code}.pkl")
        if os.path.exists(self.file_path) and not force:
            with open(self.file_path, 'rb') as file:
                self.samples = dill.load(file)
        else:
            print("No samples file found, running inference")

            guide = run_svi_inference(
                model=conditioned_locust_model,
                num_steps=num_iterations,
                verbose=True,
                lr=lr,  #0.001 worked well
                blocked_sites=["counts_obs"],
                obs_times=self.logging_times,
                data=self.subset,
                init_state=self.init_state,
                start_time=self.start_tensor,
            )

            predictive = Predictive(
            simulated_bayesian_locust, guide=guide,
            num_samples=num_samples
            )

            samples = predictive(self.init_state, 
                                self.start_tensor,
                                self.logging_times)

            with open(self.file_path, 'wb') as file:
                dill.dump(samples, file)

    def evaluate(self):
        mean_preds = {}
        abs_errors = {}
        sq_errors = {}
        self.maes = {}
        self.mses = {}
        for compartment in ["edge_l", "edge_r", "feed_l", "feed_r", "search_l", "search_r"]:
            mean_preds[compartment] = self.samples[compartment].mean(dim=0)
            abs_errors[compartment] = torch.abs(self.subset[f"{compartment}_obs"] - mean_preds[compartment])
            sq_errors[compartment] = torch.square(abs_errors[compartment])
            self.maes[compartment] = abs_errors[compartment].mean()
            self.mses[compartment] = sq_errors[compartment].mean()
            all_abs_errors = torch.cat([tensor for tensor in abs_errors.values()])
            all_sq_errors = torch.cat([tensor for tensor in sq_errors.values()])
            self.mae_mean = torch.mean(all_abs_errors).item()
            self.mse_mean = torch.mean(all_sq_errors).item()

            mean_count_overall = torch.mean(torch.concat(
            [self.subset[key] for key in self.subset.keys()]
            ))

            self.null_mse = torch.mean(torch.concat([torch.pow(torch.abs(self.subset[key] - mean_count_overall), 2) for key in self.subset.keys()]))
            self.rsquared = 1 - (self.mse_mean / self.null_mse)

    def posterior_check(self):
        fig, ax = plt.subplots(2, 3, figsize=(15, 5))
        ax = ax.flatten()

        for i, state, color in zip(
            range(6),
            ["edge_l", "edge_r", "feed_l", "feed_r", "search_l", "search_r"],
            ["green", "darkgreen", "red", "darkred", "orange", "darkorange"],
        ):
            ds_uncertainty_plot(
                state_pred=self.samples[state],
                data=self.subset[f"{state}_obs"],
                ylabel=f"# in {state}",
                color=color,
                data_label="observations",
                ax=ax[i],
                legend=True,
                test_plot=False,
                mean_label="posterior mean",
                ylim = 15
            )

    def plot_param_estimates(self, w =0, a = 0):
            plot_ds_estimates(
            self.prior_samples,
            self.samples,
            "wander",
            w,
            xlim=1,
            )

            plot_ds_estimates(
            self.prior_samples,
            self.samples,
            "attraction",
            a,
            xlim=1,
            )




    
        

        


        
    