import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pyro.infer.autoguide import AutoMultivariateNormal
import pyro


def add_ring(
    df,
    name,
    x0,
    y0,
    outside_radius=10,
    inside_radius=0,
    keep_distance=False,
    divide_by_side=False,
):
    df_c = df.copy()
    if "state" not in df.columns:
        df_c["state"] = "unclassified"

    df_c["_distance"] = np.sqrt((df_c["x"] - x0) ** 2 + (df_c["y"] - y0) ** 2)

    _condition = (
        (df_c["state"] == "unclassified")
        & (df_c["_distance"] < outside_radius)
        & (df_c["_distance"] > inside_radius)
    )
    if not divide_by_side:
        df_c.loc[_condition, "state"] = name
    else:
        df_c.loc[_condition & (df_c["x"] >= x0), "state"] = f"{name}_r"
        df_c.loc[_condition & (df_c["x"] < x0), "state"] = f"{name}_l"

    if not keep_distance:
        df_c.drop("_distance", axis=1, inplace=True)

    return df_c


def moving_average(data, window_size):
    cumsum = torch.cumsum(data, dim=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1 :] / window_size


def plot_ds_trajectories(
    data,
    times,
    window_size,
    title="Counts",
    scatter_data=None,
    scatter_data_type=None,
    observed=False,
):
    keys = ["edge_l", "edge_r", "feed_l", "feed_r", "search_l", "search_r"]
    observed_keys = [f"{key}_obs" for key in keys]
    colors = ["green", "darkgreen", "red", "darkred", "orange", "darkorange"]

    used_keys = observed_keys if observed else keys
    for state, color in zip(used_keys, colors):
        if window_size == 0:
            sns.lineplot(x=times, y=data[state], label=f"{state}", color=color)
            plt.title(title)
        else:
            running_averages = {
                state: moving_average(data[state].float(), window_size)
                for state in data
            }
            sns.lineplot(
                x=times[window_size - 1 :],
                y=running_averages[state],
                label=f"{state} (avg)",
                color=color,
            )
            plt.title(f"{title} (running average, window size = {window_size})")

    if scatter_data is not None:
        for state, color in zip(observed_keys, colors):
            sns.scatterplot(
                x=times,
                y=scatter_data[state],
                label=f"{state} - {scatter_data_type}",
                color=color,
            )

        # plt.xlim(start, end)
    plt.xlabel("time (frames)")
    plt.ylabel("# of locust")
    plt.legend(loc="upper right", fontsize="x-small")
    sns.despine()


def run_svi_inference(
    model,
    num_steps,
    verbose=True,
    lr=0.03,
    vi_family=AutoMultivariateNormal,
    guide=None,
    **model_kwargs,
):
    losses = []
    if guide is None:
        guide = vi_family(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    for step in range(1, num_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        losses.append(loss.item())
        adam.step()
        if (step % 2 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))

    return guide