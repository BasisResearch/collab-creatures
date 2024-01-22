import os
import time

import dill
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch
from chirho.dynamical.handlers import LogTrajectory, StaticBatchObservation, StaticIntervention
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import Dynamics, State, simulate
from chirho.observational.handlers import condition
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoMultivariateNormal

from collab.foraging.toolkit import run_svi_inference

pyro.settings.set(module_local_params=True)

sns.set_style("white")

# Set seed for reproducibility
seed = 123
pyro.clear_param_store()
pyro.set_rng_seed(seed)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from collab.foraging import locust as lc
from collab.foraging import toolkit as ft
from collab.utils import find_repo_root, progress_saver


def compartmentalize_locust_data(rewards, foragers, center = 50,
                                 feeding_radius = 10, edge_ring_width = 4):
    left_idx = rewards["x"].idxmin()
    right_idx = rewards["x"].idxmax()
    x_left = rewards.iloc[left_idx, 0]
    y_left = rewards.iloc[left_idx, 1]

    x_right = rewards.iloc[right_idx, 0]
    y_right = rewards.iloc[right_idx, 1]

    x_center = center
    y_center = center


    df_cat = ft.add_ring(
        foragers, "feed_l", x0=x_left, y0=y_left,
        outside_radius=feeding_radius, inside_radius=0
    )

    df_cat = ft.add_ring(
        df_cat, "feed_r", x0=x_right, y0=y_right, 
        outside_radius=feeding_radius, inside_radius=0
    )

    df_cat = ft.add_ring(
        df_cat,
        "edge",
        x0=x_center,
        y0=y_center,
        outside_radius=center + 1,
        inside_radius=center + 1 - edge_ring_width,
        divide_by_side=True,
    )

    df_cat = ft.add_ring(
        df_cat,
        "search",
        x0=x_center,
        y0=y_center,
        outside_radius=center,
        inside_radius=0,
        divide_by_side=True,
    )


    df_cat.drop(["type"], inplace=True, axis=1)

    return df_cat

    







class LocustDynamics(pyro.nn.PyroModule):
    def __init__(self, attraction, wander):
        super().__init__()
        self.attraction = attraction
        self.wander = wander

    def forward(self, X: State[torch.Tensor]):
        dX = dict()
        w_ee, w_es, w_se, w_sf, w_fs, w_ss = torch.unbind(self.wander)
        a_eler, a_erel, a_es, a_se, a_ef, a_sf, a_fs, a_slsr, a_srsl = torch.unbind(
            self.attraction
        )

        dX["edge_l"] = (
            -w_ee * X["edge_l"]  # 1-
            + w_ee * X["edge_r"]  # 2+
            - a_eler * X["edge_r"] * X["edge_l"]  # 3-
            + a_erel * X["edge_l"] * X["edge_r"]  # 4+
            - w_es * X["edge_l"]  # 5-
            - a_es * X["search_l"] * X["edge_l"]  # 6-
            + w_se * X["search_l"]  # 7+
            + a_se * X["edge_l"] * X["search_l"]  # 8+
            - a_ef * X["feed_l"] * X["edge_l"]
        )  # 9-

        dX["edge_r"] = (
            -w_ee * X["edge_r"]  # 2-
            + w_ee * X["edge_l"]  # 1+
            + a_eler * X["edge_r"] * X["edge_l"]  # 3+
            - a_erel * X["edge_l"] * X["edge_r"]  # 4-
            - w_es * X["edge_r"]  # 10-
            - a_es * X["search_r"] * X["edge_r"]  # 11-
            + w_se * X["search_r"]  # 12+
            + a_se * X["edge_r"] * X["search_r"]  # 13+
            - a_ef * X["feed_r"] * X["edge_r"]
        )  # 14-

        dX["search_l"] = (
            w_es * X["edge_l"]  # 5+
            - w_se * X["search_l"]  # 7-
            - w_sf * X["search_l"]  # 15-
            - w_ss * X["search_l"]  # 16-
            + w_ss * X["search_r"]  # 17+
            + w_fs * X["feed_l"]  # 23+
            - a_slsr * X["search_l"] * X["search_r"]  # 18-
            + a_srsl * X["search_l"] * X["search_r"]  # 19+
            + a_es * X["search_l"] * X["edge_l"]  # 6+
            - a_se * X["search_l"] * X["edge_l"]  # 8-
            + a_ef * X["feed_l"] * X["edge_l"]  # 9+
            - a_sf * X["feed_l"] * X["search_l"]  # 20-
            + a_fs * X["feed_l"] * X["search_l"]
        )  # 24+

        dX["search_r"] = (
            w_es * X["edge_r"]  # 10+
            - w_se * X["search_r"]  # 12-
            - w_sf * X["search_r"]  # 21-
            - w_ss * X["search_r"]  # 17-
            + w_ss * X["search_l"]  # 16+
            + a_slsr * X["search_l"] * X["search_r"]  # 18+
            - a_srsl * X["search_l"] * X["search_r"]  # 19-
            + a_es * X["search_r"] * X["edge_r"]  # 11+
            - a_se * X["search_r"] * X["edge_r"]  # # 13-
            + a_ef * X["feed_r"] * X["edge_r"]  # 14+
            - a_sf * X["feed_r"] * X["search_r"]  # 22-
            + w_fs * X["feed_r"]  # 25+
            + a_fs * X["feed_r"] * X["search_r"]  # 26+
        )

        dX["feed_l"] = (
            w_sf * X["search_l"]  # 15+
            + a_sf * X["feed_l"] * X["search_l"]  # 20+
            - w_fs * X["feed_l"]  # 23-
            - a_fs * X["feed_l"] * X["search_l"]
        )  # 24-

        dX["feed_r"] = (
            w_sf * X["search_r"]  # 21+
            + a_sf * X["feed_r"] * X["search_r"]  # 22+
            - w_fs * X["feed_r"]  # 25-
            - a_fs * X["feed_r"] * X["search_r"]  # 26-
        )

        return dX


def locust_noisy_model(X: State[torch.Tensor]) -> None:
    event_dim = 1 if X["edge_l"].shape and X["edge_l"].shape[-1] > 1 else 0
    keys = ["edge_l", "edge_r", "search_l", "search_r", "feed_l", "feed_r"]
    for key in keys:
        pyro.sample(f"{key}_obs", dist.Poisson(X[key]).to_event(event_dim))



def bayesian_locust(base_model=LocustDynamics) -> Dynamics[torch.Tensor]:
    with pyro.plate("attr", size=9):
        attraction = pyro.sample("attraction", dist.Uniform(0, 1))
    with pyro.plate("wond", size=6):
        wander = pyro.sample("wander", dist.Uniform(0, 1))

    locust_model = base_model(attraction, wander)
    return locust_model


def simulated_bayesian_locust(
    init_state, start_time, logging_times, base_model=LocustDynamics
) -> State[torch.Tensor]:
    locust_model = bayesian_locust(base_model)
    with TorchDiffEq(), LogTrajectory(logging_times, is_traced=True) as lt:
        simulate(locust_model, init_state, start_time, logging_times[-1])
    return lt.trajectory

def conditioned_locust(
    obs_times, data, init_state, start_time, base_model=LocustDynamics
) -> None:
    sir = bayesian_locust(base_model)
    obs = condition(data=data)(locust_noisy_model)
    with TorchDiffEq(), StaticBatchObservation(obs_times, observation=obs):
        simulate(sir, init_state, start_time, obs_times[-1])



def get_locust_posterior_samples(guide, num_samples, init_state, start_time, logging_times):
    locust_predictive = Predictive(
        simulated_bayesian_locust, guide, num_samples, init_state, start_time, logging_times)
    locust_posterior_samples = locust_predictive(init_state, start_time, logging_times)
    return locust_posterior_samples




def locust_uncertainty_plot(
    time, state_pred, ylabel, color, ax, mean_label="posterior mean"
):
    sns.lineplot(
        x=time,
        y=state_pred.mean(dim=0).squeeze().tolist(),
        color=color,
        label=mean_label,
        ax=ax,
    )
    # 95% Credible Interval
    ax.fill_between(
        time,
        torch.quantile(state_pred, 0.025, dim=0).squeeze(),
        torch.quantile(state_pred, 0.975, dim=0).squeeze(),
        alpha=0.2,
        color=color,
        label="95% credible interval",
    )

    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)

def intervention_uncertainty_plot(time_period, intervention, color, ax):
    sns.lineplot(
        x=time_period,
        y=intervention.mean(dim=0).squeeze().tolist(),
        color="grey",
        label="intervened posterior prediction",
        ax=ax)


def locust_data_plot(time, data, data_label, ax):
    sns.lineplot(x=time, y=data, color="black", ax=ax, linestyle="--", label=data_label)


def locust_test_plot(test_start_time, test_end_time, ax):
    ax.axvline(
        test_start_time, color="black", linestyle=":", label="measurement period"
    )
    ax.axvline(test_end_time, color="black", linestyle=":")


def locust_plot(
    time_period,
    state_pred,
    test_start_time,
    test_end_time,
    data,
    ylabel,
    color,
    data_label,
    ax,
    legend=False,
    test_plot=True,
    
    mean_label="posterior mean",
    xlim=None,
    intervention = None
):
    locust_uncertainty_plot(
        time_period, state_pred, ylabel, color, ax, mean_label=mean_label
    )
    locust_data_plot(time_period, data, data_label, ax)

    if intervention is not None:
        intervention_uncertainty_plot(time_period, intervention, color, ax)


    if test_plot:
        locust_test_plot(test_start_time, test_end_time, ax)
    if legend:
        ax.legend()
    else:
        ax.legend().remove()
    if xlim is not None:
        ax.set_xlim(0, xlim)
    sns.despine()




















