import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Go down one level from the current directory (replace 'folder_name' with the actual name of the folder)
folder_path = os.path.join(current_dir, "random_hungry_followers")

# Add the folder path to sys.path
sys.path.insert(0, folder_path)
# print(sys.path)

import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import pyro
import foraging_toolkit as ft
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.optim as optim
from pyro.nn import PyroModule
from pyro.infer.autoguide import (
    AutoNormal,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    init_to_mean,
    init_to_value,
)
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive
from pyro.infer import MCMC, NUTS

import os
import logging
import time

logging.basicConfig(format="%(message)s", level=logging.INFO)
smoke_test = "CI" in os.environ

import foraging_toolkit as ft


start_time = time.time()

hungry_sim = ft.Birds(
    grid_size=100, num_birds=3, num_frames=50, num_rewards=90, grab_range=3
)
hungry_sim()


hungry_sim = ft.add_hungry_birds(
    hungry_sim, num_hungry_birds=3, rewards_decay=0.3, visibility_range=6
)
end_time = time.time()

print("Generation time: ", end_time - start_time)


# ft.animate_birds(
#     hungry_sim,
#     plot_rewards=True,
#     width=600,
#     height=600,
#     point_size=10,
#     plot_traces=False,
#     trace_multiplier=4,
# )


# ft.animate_birds(
#     hungry_sim,
#     plot_rewards=True,
#     width=600,
#     height=600,
#     point_size=10,
#     plot_traces=True,
#     trace_multiplier=4,
# )
# we can also plot visibility, for one bird at a time
# as multiple birds' visibility is hard to see
# let's say, bird 2

# ft.animate_birds(
#     hungry_sim,
#     plot_rewards=True,
#     width=600,
#     height=600,
#     point_size=10,
#     plot_visibility=1,
#     plot_traces=True,
# )


start_time = time.time()
hungry_sim_derived = ft.derive_predictors(hungry_sim)
end_time = time.time()
print("derivation time:", end_time - start_time)


# ft.animate_birds(
#     hungry_sim_derived,
#     plot_rewards=True,
#     width=600,
#     height=600,
#     point_size=10,
#     plot_proximity=1,
# )


# prepare data

df = hungry_sim_derived.derivedDF.dropna()

data = torch.tensor(
    df[
        [
            "proximity_standardized",
            "trace_standardized",
            "visibility",
            "how_far_squared_scaled",
        ]
    ].values,
    # dtype=torch.float32,
)

proximity, trace, visibility, how_far = data[:, 0], data[:, 1], data[:, 2], data[:, 3]


# ft.visualise_bird_predictors(trace, proximity, how_far)


def model_sigmavar(proximity, trace, visibility, how_far):
    p = pyro.sample("p", dist.Normal(0, 0.3))
    t = pyro.sample("t", dist.Normal(0, 0.3))
    v = pyro.sample("v", dist.Normal(0, 0.3))
    b = pyro.sample("b", dist.Normal(0.5, 0.3))

    ps = pyro.sample("ps", dist.Normal(0, 0.3))
    ts = pyro.sample("ts", dist.Normal(0, 0.3))
    vs = pyro.sample("vs", dist.Normal(0, 0.3))
    bs = pyro.sample("bs", dist.Normal(0.2, 0.3))

    sigmaRaw = bs + ps * proximity + ts * trace + vs * visibility
    sigma = pyro.deterministic("sigma", F.softplus(sigmaRaw))
    mean = b + p * proximity + t * trace + v * visibility

    with pyro.plate("data", len(how_far)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=how_far)

        # Inference with SVI


# note how long this takes
# and compare with MCM

guide = AutoMultivariateNormal(model_sigmavar, init_loc_fn=init_to_mean)
svi = SVI(model_sigmavar, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

iterations = []
losses = []

pyro.clear_param_store()
num_iters = 1000
for i in range(num_iters):
    elbo = svi.step(proximity, trace, visibility, how_far)
    iterations.append(i)
    losses.append(elbo)
    if i % 200 == 0:
        logging.info("Elbo loss: {}".format(elbo))


df = pd.DataFrame({"iterations": iterations, "ELBO Loss": losses})

fig = px.line(df, x="iterations", y="ELBO Loss")

fig.update_layout(
    title="ELBO Loss over iterations",
    xaxis_title="iterations",
    yaxis_title="ELBO Loss",
    template="plotly_dark",
    showlegend=False,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
)

fig.show()


def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).transpose()
        site_stats[site_name] = describe[
            ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
        ]
    return site_stats


# inspect the summary of the SVI posterior
# of key interest: t and p

num_samples = 1000
predictive = Predictive(
    model_sigmavar, guide=guide, num_samples=num_samples, return_sites=["t", "p"]
)
random_sigmavar = {
    k: v.flatten().reshape(num_samples, -1).detach().cpu().numpy()
    for k, v in predictive(proximity, trace, visibility, how_far).items()
    if k != "obs"
}

for site, values in summary(random_sigmavar).items():
    print("Site: {}".format(site))
    print(values, "\n")
