import pandas as pd
import numpy as np
import numpyro
import copy

import plotly.express as px
import plotly.graph_objects as go
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

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)


def normalize(column):
    return (column - column.min()) / (column.max() - column.min())


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


def prep_data_for_robust_inference(sim_old, gridsize=11):
    sim_new = copy.copy(sim_old)

    def bin(vector, gridsize=11):
        vector_max = max(vector)
        vector_min = min(vector)
        # step_size = (vector_max - vector_min) / gridsize
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

    sim_new.derivedDF.dropna(inplace=True)
    sim_new.derivedDF["proximity_cat"] = bin(sim_new.derivedDF["proximity_standardized"], gridsize=gridsize)

    sim_new.derivedDF["trace_cat"] = bin(sim_new.derivedDF["trace_standardized"], gridsize=gridsize)

    sim_new.derivedDF["visibility_cat"] = bin(sim_new.derivedDF["visibility"], gridsize=gridsize)

    sim_new.derivedDF["communicate_cat"] = bin(sim_new.derivedDF["communicate_standardized"], gridsize=gridsize)

    columns_to_normalize = [
        "trace_standardized",
        "proximity_standardized",
        "communicate_standardized",
        "visibility",
        "how_far_squared_scaled",
    ]

    for column in columns_to_normalize:
        sim_new.derivedDF[column] = normalize(sim_new.derivedDF[column])

    return sim_new


def get_tensorized_data(sim_derived):
    print("Initial dataset size:", sim_derived.derivedDF.shape[0])
    df = sim_derived.derivedDF.copy().dropna()
    print("Complete cases:", df.shape[0])

    for column_name, column in df.items():
        if column.dtype.name == "category":
            df[column_name] = column.cat.codes

    trace_standardized = torch.tensor(df["trace_standardized"].values, dtype=torch.float32)
    trace_cat = torch.tensor(df["trace_cat"].values, dtype=torch.int8)
    proximity_standardized = torch.tensor(df["proximity_standardized"].values, dtype=torch.float32)
    proximity_cat = torch.tensor(df["proximity_cat"].values, dtype=torch.int8)
    visibility = torch.tensor(df["visibility"].values, dtype=torch.float32)
    visibility_cat = torch.tensor(df["visibility_cat"].values, dtype=torch.int8)
    communicate_standardized = torch.tensor(df["communicate_standardized"].values, dtype=torch.float32)
    communicate_cat = torch.tensor(df["communicate_cat"].values, dtype=torch.int8)
    how_far_squared_scaled = torch.tensor(df["how_far_squared_scaled"].values, dtype=torch.float32)

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


def get_svi_results(df):
    def discretized_p(proximity_id, how_far):
        p = numpyro.sample("p", dist.Normal(0, 0.5).expand([len(set(proximity_id))]))
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        mu = p[proximity_id]
        numpyro.sample("how_far", dist.Normal(mu, sigma), obs=how_far)

    def discretized_t(trace_id, how_far):
        t = numpyro.sample("t", dist.Normal(0, 0.5).expand([len(set(trace_id))]))
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        mu = t[trace_id]
        numpyro.sample("how_far", dist.Normal(mu, sigma), obs=how_far)

    def discretized_c(communicate_id, how_far):
        c = numpyro.sample("c", dist.Normal(0, 0.5).expand([len(set(communicate_id))]))
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        mu = c[communicate_id]
        numpyro.sample("how_far", dist.Normal(mu, sigma), obs=how_far)

    guide_p = AutoLaplaceApproximation(discretized_p)
    guide_t = AutoLaplaceApproximation(discretized_t)
    guide_c = AutoLaplaceApproximation(discretized_c)

    svi_p = SVI(
        discretized_p,
        guide_p,
        optim.Adam(1),
        Trace_ELBO(),
        proximity_id=df.proximity_id.values,
        how_far=df.how_far.values,
    )
    svi_t = SVI(
        discretized_t,
        guide_t,
        optim.Adam(1),
        Trace_ELBO(),
        trace_id=df.trace_id.values,
        how_far=df.how_far.values,
    )
    svi_c = SVI(
        discretized_c,
        guide_c,
        optim.Adam(1),
        Trace_ELBO(),
        communicate_id=df.communicate_id.values,
        how_far=df.how_far.values,
    )

    svi_result_p = svi_p.run(random.PRNGKey(0), 2000)
    svi_result_t = svi_t.run(random.PRNGKey(0), 2000)
    svi_result_c = svi_c.run(random.PRNGKey(0), 2000)

    return svi_result_p, svi_result_t, svi_result_c


def sample_and_plot_coef(coef, input, model):
    coef_samples = []
    for _ in range(1000):
        X_resampled, y_resampled = resample(input, summary[f"params_{coef}"], random_state=np.random.randint(1000))

        model.fit(X_resampled, y_resampled)
        coef_samples.append(model.coef_[0])

    histogram_trace = go.Histogram(
        x=coef_samples,
        marker=dict(color="blue"),
    )

    layout = go.Layout(
        title=f"Histogram of coef_samples_{coef}",
        xaxis=dict(title="Coefficient Value"),
        yaxis=dict(title="Frequency"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
    )

    fig = go.Figure(data=[histogram_trace], layout=layout)
    fig.show()


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
            describe = marginal_site.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose()
            site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
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
        for k, v in predictive(proximity, trace, visibility, communicate, how_far_score).items()
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
