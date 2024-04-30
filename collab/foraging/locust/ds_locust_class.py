import os

import dill
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from chirho.dynamical.handlers import LogTrajectory
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import simulate
from pyro.infer import Predictive

from collab.foraging.toolkit.dynamical_utils import (
    ds_uncertainty_plot,
    plot_ds_trajectories,
    run_svi_inference,
)
from collab.utils import find_repo_root

from .ds_locust_functions import (
    LocustDynamics,
    conditioned_locust_model,
    get_count_data_subset,
    plot_ds_estimates,
    simulated_bayesian_locust,
)

root = find_repo_root()


class LocustDS:
    def __init__(self, data_code, start, end):
        self.data_code = data_code
        self.start = start
        self.start_tensor = torch.tensor(self.start).float()
        self.end = end
        self.end_tensor = torch.tensor(self.end).float()

        self.root = find_repo_root()
        self.data_path = os.path.join(
            self.root, f"data/foraging/locust/ds/locust_counts{data_code}.pkl"
        )

        with open(self.data_path, "rb") as f:
            locust_count_data = dill.load(f)

            self.count_data = locust_count_data["count_data"]

        self.logging_times = torch.arange(start, end, 1)
        self.c_data = get_count_data_subset(self.count_data, self.start, self.end)
        self.subset = self.c_data["count_subset"]

        # TODO make sure this works properly
        # assert self.subset['edge_l_obs'].shape[0] == self.end - self.start

        self.init_state = self.c_data["init_state"]
        self.N = torch.sum((torch.stack(list(self.init_state.values()))))

        self.piecemeal_path = os.path.join(self.root, "data/foraging/locust/ds/")
        self.validation = {}

    def simulate_trajectories(self, true_attraction, true_wander, init_state=None):

        locust_true = LocustDynamics(true_attraction, true_wander)
        with TorchDiffEq(
            method="rk4",
        ), LogTrajectory(self.logging_times) as lt:
            simulate(locust_true, self.init_state, self.start_tensor, self.end_tensor)

        self.simulated_traj = lt.trajectory

    def plot_simulated_trajectories(self, window_size=0):
        plot_ds_trajectories(
            self.simulated_traj, self.logging_times, window_size=window_size
        )

    def get_prior_samples(self, num_samples, force=False):
        self.priors_path = os.path.join(
            self.piecemeal_path,
            f"priors_sam{num_samples}_{self.data_code}_s{self.start}_e{self.end}.pkl",
        )
        if os.path.exists(self.priors_path) and not force:
            with open(self.priors_path, "rb") as file:
                self.prior_samples = dill.load(file)

        else:
            prior_predictive = Predictive(
                simulated_bayesian_locust, num_samples=num_samples
            )
            prior_samples = prior_predictive(
                self.init_state, self.start_tensor, self.logging_times
            )
            self.prior_samples = prior_samples

            with open(self.priors_path, "wb") as file:
                dill.dump(prior_samples, file)

        for key in self.init_state.keys():
            assert (
                self.init_state[key].item() == self.prior_samples[key][0, 0, 0].item()
            ), "prior predictive inits are wrong"

    def run_inference(
        self, name, num_iterations, lr=0.001, num_samples=150, force=False, save=False
    ):
        self.file_path = os.path.join(
            self.piecemeal_path,
            f"{name}_s{self.start}_e{self.end}_i{num_iterations}_{self.data_code}.pkl",
        )
        if os.path.exists(self.file_path) and not force:
            print("Loading inference samples")
            with open(self.file_path, "rb") as file:
                self.samples = dill.load(file)
        else:
            print("Running inference")

            self.guide = run_svi_inference(
                model=conditioned_locust_model,
                num_steps=num_iterations,
                verbose=True,
                lr=lr,  # 0.001 worked well
                blocked_sites=["counts_obs"],
                obs_times=self.logging_times,
                data=self.subset,
                init_state=self.init_state,
                start_time=self.start_tensor,
            )

            self.predictive = Predictive(
                simulated_bayesian_locust, guide=self.guide, num_samples=num_samples
            )

            self.samples = self.predictive(
                self.init_state, self.start_tensor, self.logging_times
            )

            if save:
                with open(self.file_path, "wb") as new_file:
                    dill.dump(self.samples, new_file)

        for key in self.init_state.keys():
            assert (
                self.init_state[key].item() == self.samples[key][0, 0, 0].item()
            ), "predictive inits are wrong"

    def evaluate(self, samples, subset, check=True, figure=True, plot_null_model=True):

        uniform_preds = {}
        uniform_residuals = {}
        uniform_abs_errors = {}
        uniform_sq_errors = {}

        mean_preds = {}
        residuals = {}
        abs_errors = {}
        sq_errors = {}

        compartment_colors = {
            "edge_l": "green",
            "edge_r": "darkgreen",
            "feed_l": "red",
            "feed_r": "darkred",
            "search_l": "orange",
            "search_r": "darkorange",
        }

        if figure:
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        for i, compartment in enumerate(compartment_colors.keys()):

            mean_preds[compartment] = samples[compartment].mean(dim=0)
            # ran into a bug here before, hence assertion
            assert samples[compartment][0, 0, 0] == mean_preds[compartment][0][0]

            residuals[compartment] = (
                mean_preds[compartment] - subset[f"{compartment}_obs"]
            )
            abs_errors[compartment] = torch.abs(residuals[compartment])
            sq_errors[compartment] = torch.square(abs_errors[compartment])

            uniform_preds[compartment] = (self.N / 6).expand(
                mean_preds[compartment].shape
            )

            assert uniform_preds[compartment].shape == mean_preds[compartment].shape

            uniform_residuals[compartment] = (
                uniform_preds[compartment] - subset[f"{compartment}_obs"]
            )
            uniform_abs_errors[compartment] = torch.abs(uniform_residuals[compartment])
            uniform_sq_errors[compartment] = torch.square(
                uniform_abs_errors[compartment]
            )

            if figure:
                row = i // 3
                col = i % 3
                ax = axs[row, col]

                if plot_null_model:
                    ax.scatter(
                        y=uniform_residuals[compartment].flatten(),
                        x=self.logging_times,
                        color="grey",
                        label="null model",
                    )

                ax.scatter(
                    y=residuals[compartment].flatten(),
                    x=self.logging_times,
                    color=compartment_colors[compartment],
                    label=compartment,
                )

                ax.set_title(f"{compartment}")

        all_sq_errors = torch.cat([tensor for tensor in sq_errors.values()])
        all_abs_errors = torch.cat([tensor for tensor in abs_errors.values()])
        mse = torch.mean(all_sq_errors).item()
        mae = torch.mean(all_abs_errors).item()

        all_uniform_sq_errors = torch.cat(
            [tensor for tensor in uniform_sq_errors.values()]
        )
        all_uniform_abs_errors = torch.cat(
            [tensor for tensor in uniform_abs_errors.values()]
        )

        uniform_mse = torch.mean(all_uniform_sq_errors).item()
        uniform_mae = torch.mean(all_uniform_abs_errors).item()
        rsqared = 1 - mse / uniform_mse

        if figure:
            fig.suptitle(
                f"Residuals vs. time (overall mae: {mae:.2f}, "
                f"null mae: {uniform_mae:.2f}, $R^2$: {rsqared:.3f})"
            )
            plt.xlabel("time")
            plt.ylabel("residuals")
            plt.legend()
            sns.despine()
            plt.show()

        if check:
            self.mae = mae
            self.null_mae = uniform_mae
            self.rsquared = rsqared

        else:
            return {"mae": mae, "null mae": uniform_mae, "rsquared": rsqared}

    def posterior_check(self, samples=None, subset=None, title=None, save=False):
        if title is None:
            title = f"Posterior predictive check ({self.start * 10} to {self.end * 10})"
        if samples is None:
            samples = self.samples
        if subset is None:
            subset = self.subset

        fig, ax = plt.subplots(2, 3, figsize=(15, 5))
        ax = ax.flatten()

        for i, state, color in zip(
            range(6),
            ["edge_l", "edge_r", "feed_l", "feed_r", "search_l", "search_r"],
            ["green", "darkgreen", "red", "darkred", "orange", "darkorange"],
        ):
            ds_uncertainty_plot(
                state_pred=samples[state],
                data=subset[f"{state}_obs"],
                ylabel=f"# in {state}",
                color=color,
                data_label="observations",
                ax=ax[i],
                legend=True,
                test_plot=False,
                mean_label="posterior mean",
                ylim=15,
            )

        fig.suptitle(title)

        if save:
            path = os.path.join(root, "docs/figures/locust_posterior_check.png")
            fig.savefig(path)

    def plot_param_estimates(self, w=0, a=0, xlim=1, save=False):
        plot_ds_estimates(
            self.prior_samples, self.samples, "wander", w, xlim=xlim, save=save
        )

        plot_ds_estimates(
            self.prior_samples, self.samples, "attraction", a, xlim=xlim, save=save
        )

    def validate(
        self,
        validation_data_code,
        num_iterations=1500,
        lr=0.001,
        num_samples=150,
        force=False,
        save=False,
        name="length",
    ):
        self.v_data_path = os.path.join(
            self.root,
            f"data/foraging/locust/ds/locust_counts{validation_data_code}.pkl",
        )

        with open(self.v_data_path, "rb") as f:
            validation_data = dill.load(f)

        self.v_count_data = validation_data["count_data"]

        self.v_data = get_count_data_subset(self.v_count_data, self.start, self.end)

        self.v_subset = self.v_data["count_subset"]
        self.v_init_state = self.v_data["init_state"]

        self.v_file_path = os.path.join(
            self.piecemeal_path,
            f"v_{name}_s{self.start}_e{self.end}_i{num_iterations}_{self.data_code}_v{validation_data_code}.pkl",
        )
        if os.path.exists(self.v_file_path) and not force:
            print("Loading validation samples")
            with open(self.v_file_path, "rb") as v_file:
                self.v_samples = dill.load(v_file)
        else:
            print("Generating validation samples")
            if not hasattr(self, "guide"):
                print("Running inference for validation")
                self.guide = run_svi_inference(
                    model=conditioned_locust_model,
                    num_steps=num_iterations,
                    verbose=True,
                    lr=lr,  # 0.001 worked well
                    blocked_sites=["counts_obs"],
                    obs_times=self.logging_times,
                    data=self.subset,  # train on original data
                    init_state=self.init_state,
                    start_time=self.start_tensor,
                )

                self.predictive = Predictive(
                    simulated_bayesian_locust, guide=self.guide, num_samples=num_samples
                )

            self.v_samples = self.predictive(
                self.v_init_state, self.start_tensor, self.logging_times
            )

            if save:
                with open(self.v_file_path, "wb") as new_v_file:
                    dill.dump(self.v_samples, new_v_file)

        for key in self.v_init_state.keys():
            assert (
                self.v_init_state[key].item() == self.v_samples[key][0, 0, 0].item()
            ), "validation inits are wrong"

        self.validation[validation_data_code] = self.evaluate(
            samples=self.v_samples, subset=self.v_subset, check=False
        )
