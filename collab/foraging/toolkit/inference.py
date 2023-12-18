import copy

import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
import pandas as pd
import torch
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


def normalize(column):
    return (column - column.min()) / (column.max() - column.min())


def prep_data_for_robust_inference(sim_old, gridsize=11):
    sim_new = copy.copy(sim_old)

    def bin_vector(vector, gridsize=11):
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
    sim_new.derivedDF["proximity_cat"] = bin_vector(
        sim_new.derivedDF["proximity_standardized"], gridsize=gridsize
    )

    sim_new.derivedDF["trace_cat"] = bin_vector(
        sim_new.derivedDF["trace_standardized"], gridsize=gridsize
    )

    sim_new.derivedDF["visibility_cat"] = bin_vector(
        sim_new.derivedDF["visibility"], gridsize=gridsize
    )

    sim_new.derivedDF["communicate_cat"] = bin_vector(
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

    sim_new_df = sim_new.derivedDF

    sim_new_df["proximity_id"] = sim_new_df.proximity_cat.astype("category").cat.codes
    sim_new_df["trace_id"] = sim_new_df.trace_cat.astype("category").cat.codes
    sim_new_df["communicate_id"] = sim_new_df.communicate_cat.astype(
        "category"
    ).cat.codes
    sim_new_df["how_far"] = sim_new_df.how_far_squared_scaled

    sim_new.derivedDF = sim_new_df
    return sim_new_df


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
