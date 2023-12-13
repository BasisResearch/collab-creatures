import copy
import logging

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import torch
import torch.nn.functional as F
from jax import random
from numpyro.diagnostics import print_summary
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    init_to_mean,
    init_to_value,
)
from pyro.nn import PyroModule
from pyro.optim import Adam
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

logging.basicConfig(format="%(message)s", level=logging.INFO)


def normalize(column):
    return (column - column.min()) / (column.max() - column.min())


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
    sim_new.derivedDF["proximity_cat"] = bin(
        sim_new.derivedDF["proximity_standardized"], gridsize=gridsize
    )

    sim_new.derivedDF["trace_cat"] = bin(
        sim_new.derivedDF["trace_standardized"], gridsize=gridsize
    )

    sim_new.derivedDF["visibility_cat"] = bin(
        sim_new.derivedDF["visibility"], gridsize=gridsize
    )

    sim_new.derivedDF["communicate_cat"] = bin(
        sim_new.derivedDF["communicate_standardized"], gridsize=gridsize
    )

    columns_to_normalize = [
        "trace_standardized",
        "proximity_standardized",
        "communicate_standardized",
        "visibility",
        "how_far_squared_scaled",
    ]

    for column in columns_to_normalize:
        sim_new.derivedDF[column] = normalize(sim_new.derivedDF[column])

    sim_new_DF = sim_new.derivedDF

    # TODO work this in with debugging
    sim_new_DF["proximity_id"] = sim_new_DF.proximity_cat.astype("category").cat.codes
    sim_new_DF["trace_id"] = sim_new_DF.trace_cat.astype("category").cat.codes
    sim_new_DF["communicate_id"] = sim_new_DF.communicate_cat.astype(
        "category"
    ).cat.codes
    sim_new_DF["how_far"] = sim_new_DF.how_far_squared_scaled

    sim_new.derivedDF = sim_new_DF
    return sim_new_DF


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
