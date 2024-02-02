import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from collab.utils import find_repo_root


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
        & (df_c["_distance"] <= outside_radius)
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


def tensorize_and_dump_count_data(df_cat, data_path):
    root = find_repo_root()
    all_states = df_cat["state"].unique()
    times = sorted(df_cat["time"].unique())
    N_obs = max(times)
    all_combinations = pd.DataFrame(
        [(state, time) for state in all_states for time in times],
        columns=["state", "time"],
    )
    counts = pd.merge(
        all_combinations,
        df_cat.groupby(["state", "time"]).size().reset_index(name="count"),
        how="left",
        on=["state", "time"],
    )
    counts["count"].fillna(0, inplace=True)

    count_data = {}
    for state in all_states:
        count_data[f"{state}_obs"] = torch.tensor(
            counts[counts["state"] == state]["count"].values
        )

    shapes = [tensor.shape for tensor in count_data.values()]
    assert all(shape == shapes[0] for shape in shapes)

    tensor_length = len(next(iter(count_data.values())))

    assert (
        N_obs == tensor_length
    ), "Tensor length does not match number of observations!"

    sums_per_position = [
        sum(count_data[state][k] for state in count_data) for k in range(tensor_length)
    ]
    assert all(
        sums_per_position[0] == sum_at_k for sum_at_k in sums_per_position[1:]
    ), "Population count is not constant!"

    init_state = {key[:-4]: count_data[key][0] for key in count_data.keys()}

    tensorized_count_data = {"count_data": count_data, "init_state": init_state}
    tensorized_count_data_path = os.path.join(root, data_path)
    with open(tensorized_count_data_path, "wb") as f:
        dill.dump(tensorized_count_data, f)

    return tensorized_count_data


def moving_average(data, window_size):
    cumsum = torch.cumsum(data, dim=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1 :] / window_size


def plot_ds_trajectories(
    data,
    times=None,
    window_size=0,
    title="Counts",
    scatter_data=None,
    scatter_data_type=None,
    observed=False,
    keys=None,
    observed_keys=None,
    colors=None,
):
    if times is None:
        first_tensor = next(iter(data.values()))
        times = torch.arange(len(first_tensor))

    # default for locust, consider factoring out later
    if keys is None:
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


def ds_predictive_plot(
    state_pred=None,
    time=None,
    ylabel=None,
    color=None,
    ax=None,
    mean_label="posterior mean",
):
    sns.lineplot(
        x=time,
        y=state_pred.mean(dim=0).squeeze().tolist(),
        color=color,
        label=mean_label,
        ax=ax,
    )
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


def ds_intervention_plot(time, intervention, ax):
    sns.lineplot(
        x=time,
        y=intervention.mean(dim=0).squeeze().tolist(),
        color="grey",
        label="intervened posterior prediction",
        ax=ax,
    )


def ds_data_plot(data=None, time=None, data_label=None, ax=None):
    sns.lineplot(x=time, y=data, color="black", ax=ax, linestyle="--", label=data_label)


def ds_test_plot(test_start_time, test_end_time, ax):
    ax.axvline(
        test_start_time, color="black", linestyle=":", label="measurement period"
    )
    ax.axvline(test_end_time, color="black", linestyle=":")


def ds_uncertainty_plot(
    state_pred,
    ylabel,
    color,
    ax,
    data=None,
    data_label="observations",
    time=None,
    legend=False,
    test_plot=True,
    test_start_time=None,
    test_end_time=None,
    mean_label="posterior mean",
    xlim=None,
    ylim = None,
    intervention=None,
):
    if time is None:
        time = torch.arange(state_pred.shape[2])

    ds_predictive_plot(
        state_pred=state_pred,
        time=time,
        ylabel=ylabel,
        color=None,
        ax=ax,
        mean_label=mean_label,
    )
    if data is not None:
        ds_data_plot(data=data, time=time, data_label=data_label, ax=ax)

    if intervention is not None:
        ds_intervention_plot(time, intervention, color, ax)

    if test_plot:
        ds_test_plot(test_start_time, test_end_time, ax)
    if legend:
        ax.legend()
    else:
        ax.legend().remove()
    if xlim is not None:
        ax.set_xlim(0, xlim)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    sns.despine()


def run_svi_inference(
    model,
    num_steps=500,
    verbose=True,
    lr=0.03,
    guide=None,
    blocked_sites=None,
    **model_kwargs,
):
    losses = []
    running_loss_means = []
    if guide is None:
        guide = vi_family = AutoMultivariateNormal(
            pyro.poutine.block(model, hide=blocked_sites)
        )
    elbo = pyro.infer.Trace_ELBO()(model, guide)

    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    print(f"Running SVI for {num_steps} steps...")
    for step in range(1, num_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        losses.append(loss.item())
        running_loss_means = [np.nan] * 30
        if step > 31:
            running_loss_mean = np.mean(losses[-30:])
            running_loss_means.append(running_loss_mean)
        adam.step()
        if (step % 50 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))

    plt.plot(losses, label='ELBO loss')
    plt.plot(running_loss_means, label='running mean', color='gray', linestyle='--')
    sns.despine()
    plt.title("ELBO Loss")
    plt.ylim(0, max(losses))
    plt.legend()
    plt.show()

    return guide
