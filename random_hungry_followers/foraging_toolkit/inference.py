import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch
import pyro.optim as optim
from pyro.nn import PyroModule
from pyro.infer.autoguide import (
    AutoNormal,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    init_to_mean,
    init_to_value,
)
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive
from pyro.infer import MCMC, NUTS
import torch.nn.functional as F


import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)


def prep_data_for_communicators_inference(sim_derived):
    print("Initial dataset size:", sim_derived.derivedDF.shape[0])
    # df = sim_derived.derivedDF.copy().dropna()
    print("Complete cases:", df.shape[0])
    # large drop expected as we only care about points within birds' visibility range
    # and many communicates are outside of it

    data = torch.tensor(
        df[
            [
                "trace_standardized",
                "proximity_standardized",
                "visibility",
                "communicate_standardized",
                "how_far_squared_scaled",
            ]
        ].values,
        dtype=torch.float32,
    )

    trace, proximity, visibility, communicate, how_far = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
    )

    print(str(len(proximity)) + " data points prepared for inference.")

    return trace, proximity, visibility, communicate, how_far


def prep_data_for_robust_inference(sim, gridsize=11):
    def bin(vector, gridsize=11):
        vector_max = max(vector)
        vector_min = min(vector)
        step_size = (vector_max - vector_min) / gridsize
        vector_bin_edges = np.linspace(vector_min, vector_max, gridsize + 1)
        # vector_bin_edges = pd.interval_range(start=vector_min, end=vector_max, freq=step_size)
        vector_bin_labels = [f"{i}" for i in range(1, gridsize + 1)]
        vector_binned = pd.cut(
            vector,
            bins=vector_bin_edges,
            labels=vector_bin_labels,
            include_lowest=True,
        )

        return vector_binned

    def normalize(column):
        return (column - column.min()) / (column.max() - column.min())

    sim.derivedDF = sim.derivedDF.dropna(inplace=True)
    sim.derivedDF["proximity_cat"] = bin(
        sim.derivedDF["proximity_standardized"], gridsize=gridsize
    )

    sim.derivedDF["trace_cat"] = bin(
        sim.derivedDF["trace_standardized"], gridsize=gridsize
    )

    sim.derivedDF["visibility_cat"] = bin(
        sim.derivedDF["visibility"], gridsize=gridsize
    )

    sim.derivedDF["communicate_cat"] = bin(
        sim.derivedDF["communicate_standardized"], gridsize=gridsize
    )

    columns_to_normalize = [
        "trace_standardized",
        "proximity_standardized",
        "communicate_standardized",
        "visibility",
        "how_far_squared_scaled",
    ]

    for column in columns_to_normalize:
        sim.derivedDF[column] = normalize(sim.derivedDF[column])

    return sim


def get_tensorized_data(sim_derived):
    print("Initial dataset size:", sim_derived.derivedDF.shape[0])
    df = sim_derived.derivedDF.copy().dropna()
    print("Complete cases:", df.shape[0])

    for column_name, column in df.items():
        if column.dtype.name == "category":
            df[column_name] = column.cat.codes

    trace_standardized = torch.tensor(
        df["trace_standardized"].values, dtype=torch.float32
    )
    trace_cat = torch.tensor(df["trace_cat"].values, dtype=torch.int8)
    proximity_standardized = torch.tensor(
        df["proximity_standardized"].values, dtype=torch.float32
    )
    proximity_cat = torch.tensor(df["proximity_cat"].values, dtype=torch.int8)
    visibility = torch.tensor(df["visibility"].values, dtype=torch.float32)
    visibility_cat = torch.tensor(df["visibility_cat"].values, dtype=torch.int8)
    communicate_standardized = torch.tensor(
        df["communicate_standardized"].values, dtype=torch.float32
    )
    communicate_cat = torch.tensor(df["communicate_cat"].values, dtype=torch.int8)
    how_far_squared_scaled = torch.tensor(
        df["how_far_squared_scaled"].values, dtype=torch.float32
    )

    data = {
        "trace_standardized": trace_standardized,
        "trace_cat": trace_cat,
        "proximity_standardized": proximity_standardized,
        "proximity_cat": proximity_cat,
        "visibility": visibility,
        "visibility_cat": visibility_cat,
        "communicate_standardized": communicate_standardized,
        "communicate_cat": communicate_cat,
        "how_far": how_far_squared_scaled,
    }

    return data


def model_sigmavar_com(proximity, trace, visibility, communicate, how_far_score):
    p = pyro.sample("p", dist.Normal(0, 0.3))
    t = pyro.sample("t", dist.Normal(0, 0.3))
    v = pyro.sample("v", dist.Normal(0, 0.3))
    c = pyro.sample("c", dist.Normal(0, 0.3))
    b = pyro.sample("b", dist.Normal(0.5, 0.3))

    ps = pyro.sample("ps", dist.Normal(0, 0.3))
    ts = pyro.sample("ts", dist.Normal(0, 0.3))
    vs = pyro.sample("vs", dist.Normal(0, 0.3))
    cs = pyro.sample("cs", dist.Normal(0, 0.3))
    bs = pyro.sample("bs", dist.Normal(0.2, 0.3))

    sigmaRaw = bs + ps * proximity + ts * trace + vs * visibility + cs * communicate
    sigma = pyro.deterministic("sigma", F.softplus(sigmaRaw))
    mean = b + p * proximity + t * trace + v * visibility + c * communicate

    with pyro.plate("data", len(how_far_score)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=how_far_score)


def svi_training(model, proximity, trace, visibility, communicate, how_far_score):
    guide = AutoMultivariateNormal(model_sigmavar_com, init_loc_fn=init_to_mean)
    svi = SVI(model, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

    iterations = []
    losses = []

    pyro.clear_param_store()
    num_iters = 1000
    for i in range(num_iters):
        elbo = svi.step(proximity, trace, visibility, communicate, how_far_score)
        iterations.append(i)
        losses.append(elbo)
        if i % 200 == 0:
            logging.info("Elbo loss: {}".format(elbo))

    return guide


def summary(samples, sites=None):
    if sites is None:
        sites = [site_name for site_name in samples.keys()]

    site_stats = {}
    for site_name, values in samples.items():
        if site_name in sites:
            marginal_site = pd.DataFrame(values)
            describe = marginal_site.describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).transpose()
            site_stats[site_name] = describe[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
    return site_stats


def svi_prediction(
    model,
    guide,
    proximity,
    trace,
    visibility,
    communicate,
    how_far_score,
    num_samples=1000,
):
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        return_sites=["t", "p", "c"],
    )

    communicate_sigmavar = {
        k: v.flatten().reshape(num_samples, -1).detach().cpu().numpy()
        for k, v in predictive(
            proximity, trace, visibility, communicate, how_far_score
        ).items()
        if k != "obs"
    }

    for site, values in summary(communicate_sigmavar, ["t", "p", "c"]).items():
        print("Site: {}".format(site))
        print(values, "\n")

    return communicate_sigmavar


def mcmc_training(model, num_samples, sites=None, *args):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_samples // 4)
    mcmc.run(*args)

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    if sites is None:
        sites = [site_name for site_name in hmc_samples.keys()]

    for site, values in summary(hmc_samples).items():
        if site in ["t", "p"]:
            print("Site: {}".format(site))
            print(values, "\n")

    return hmc_samples
