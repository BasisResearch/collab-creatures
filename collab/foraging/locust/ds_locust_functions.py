import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch
from chirho.dynamical.handlers import LogTrajectory, StaticBatchObservation
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import Dynamics, State, simulate
from chirho.observational.handlers import condition
from pyro.infer import Predictive

from collab.foraging import toolkit as ft

pyro.settings.set(module_local_params=True)

sns.set_style("white")

# Set seed for reproducibility
seed = 123
pyro.clear_param_store()
pyro.set_rng_seed(seed)


def compartmentalize_locust_data(
    rewards, foragers, center=50, feeding_radius=10, edge_ring_width=4
):
    left_idx = rewards["x"].idxmin()
    right_idx = rewards["x"].idxmax()
    x_left = rewards.iloc[left_idx, 0]
    y_left = rewards.iloc[left_idx, 1]

    x_right = rewards.iloc[right_idx, 0]
    y_right = rewards.iloc[right_idx, 1]

    x_center = center
    y_center = center

    df_cat = ft.add_ring(
        foragers,
        "feed_l",
        x0=x_left,
        y0=y_left,
        outside_radius=feeding_radius,
        inside_radius=0,
    )

    df_cat = ft.add_ring(
        df_cat,
        "feed_r",
        x0=x_right,
        y0=y_right,
        outside_radius=feeding_radius,
        inside_radius=0,
    )

    df_cat = ft.add_ring(
        df_cat,
        "edge",
        x0=x_center,
        y0=y_center,
        outside_radius=center + 3,
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


def get_count_data_subset(count_data, start, end):
    count_subset = {key: count_data[key][start:end] for key in count_data.keys()}
    init_state = {key[:-4]: count_subset[key][0] for key in count_subset.keys()}

    return {"count_subset": count_subset, "init_state": init_state}


class LocustDynamics(pyro.nn.PyroModule):
    def __init__(self, attraction, wander):
        super().__init__()
        self.attraction = attraction
        self.wander = wander

    def forward(self, X: State[torch.Tensor]):
        dX = dict()

        w_sides, w_inside, w_outside, w_feed = torch.unbind(self.wander)        
        a_r, a_l, a_edge, a_search, a_feed = torch.unbind(self.attraction)

        epsilon = 0.0001

        w_edgers_lr = w_sides * X['edge_l']
        w_edgers_rl = w_sides * X['edge_r']

        a_edgers_lr = a_r * X['edge_l'] * X['edge_r']
        a_edgers_rl = a_l * X['edge_r'] * X['edge_l']

        w_edgers_ls = w_inside * X['edge_l']
        w_edgers_rs = w_inside * X['edge_r']

        a_edgers_ls = a_search * X['edge_l'] * X['search_l']
        a_edgers_rs = a_search * X['edge_r'] * X['search_r']

        a_edgers_lf = a_feed * X['edge_l'] * X['feed_l']
        a_edgers_rf = a_feed * X['edge_r'] * X['feed_r']

        w_searchers_le = w_outside * X['search_l']
        w_searchers_re = w_outside * X['search_r']

        a_searchers_le = a_edge * X['search_l'] * X['edge_l']
        a_searchers_re = a_edge * X['search_r'] * X['edge_r']

        w_searchers_lr = w_sides * X['search_l']
        w_searchers_rl = w_sides * X['search_r']

        a_searchers_lr = a_r * X['search_l'] * X['search_r']
        a_searchers_rl = a_l * X['search_r'] * X['search_l']


        w_searchers_lf = w_feed * X['search_l']
        w_searchers_rf = w_feed * X['search_r']

        a_searchers_lf = a_feed * X['search_l'] * X['feed_l']
        a_searchers_rf = a_feed * X['search_r'] * X['feed_r']

        w_feeders_l = w_outside * X['feed_l'] 
        w_feeders_r = w_outside * X['feed_r']

        a_feeders_l = a_search * X['feed_l'] * X['search_l']
        a_feeders_r = a_search * X['feed_r'] * X['search_r']

        dX["edge_l"] = (
            - w_edgers_lr  # 1-
            + w_edgers_rl  # 2+
            - a_edgers_lr  # 3-
            + a_edgers_rl  # 4+
            - w_edgers_ls  # 5-
            - a_edgers_ls  # 7-
            - a_edgers_lf  # 9-
            + w_searchers_le # 11+
            + a_searchers_le #  13+
        ) + epsilon

        dX["edge_r"] = (
            - w_edgers_rl  # 2-
            + w_edgers_lr  # 1+
            + a_edgers_lr  # 3+
            - a_edgers_rl  # 4-
            - w_edgers_rs  # 6-
            - a_edgers_rs  # 8-
            - a_edgers_rf  # 10-
            + w_searchers_re # 12+
            + a_searchers_re # 14+
        ) + epsilon

        dX['search_l'] = (
            w_edgers_ls  # 5+
            + a_edgers_ls  # 7+
            + a_edgers_lf  #  9+
            - w_searchers_le # 11-
            - w_searchers_lr # 15-
            + w_searchers_rl # 16+
            - w_searchers_lf # 19-
            - a_searchers_lr # 17-
            + a_searchers_rl # 18+
            - a_searchers_le # 13-
            - a_searchers_lf # 21-
            + w_feeders_l # 23+
            + a_feeders_l # 25+
        ) + epsilon

        dX['search_r'] = (
            w_edgers_rs  # 6+
            + a_edgers_rs  # 8+
            + a_edgers_rf  # 10+
            - w_searchers_re # 12-
            - w_searchers_rl # 16-
            + w_searchers_lr # 15+
            - w_searchers_rf # 20-
            - a_searchers_rl # 18-
            + a_searchers_lr # 17+
            - a_searchers_re # 14-
            - a_searchers_rf # 22-
            + w_feeders_r # 24+
            + a_feeders_r # 26+
        ) + epsilon

        dX['feed_l'] = (
            w_searchers_lf  # 19+
            + a_searchers_lf  # 21+
            - w_feeders_l # 23-
            - a_feeders_l # 25-
        ) + epsilon


        dX['feed_r'] = (
            w_searchers_rf  # 20+ 
            + a_searchers_rf  # 22+
            - w_feeders_r # 24-
            - a_feeders_r # 26-
        ) + epsilon

        return dX
        

def bayesian_locust(base_model=LocustDynamics) -> Dynamics[torch.Tensor]:
    with pyro.plate("attr", size=5):
        attraction = pyro.sample("attraction",  dist.Uniform(0.00001,.1)) # dist.LogNormal(.3, 8))
    with pyro.plate("wond", size=4):
        wander = pyro.sample("wander", dist.Uniform(0.00001, .3)) #dist.LogNormal(.5, 5))

    locust_model = base_model(attraction, wander)
    return locust_model

def locust_noisy_model(X: State[torch.Tensor]) -> None:
    keys = ["edge_l", "edge_r", "search_l", "search_r", "feed_l", "feed_r"]

    counts = torch.stack([X[key] for key in keys], dim=-1)
    total_count = int(torch.sum(counts[0], dim=-1, keepdim=True))

    probs = (counts / total_count).clamp(min=0.0001, max=0.9999)
    probs = probs / probs.sum(-1, keepdim=True)

    with pyro.plate("data", len(X["edge_l"])):
        pyro.sample(
            "counts_obs", dist.Multinomial(total_count, probs=probs)#.to_event(event_dim)
        )

def conditioned_locust_model(
    obs_times, data, init_state, start_time, base_model=LocustDynamics
) -> None:
    bayesian = bayesian_locust(base_model)
    obs = condition(data=data)(locust_noisy_model)
    with TorchDiffEq(method='rk4'), StaticBatchObservation(obs_times, observation=obs):
        simulate(bayesian, init_state, start_time, obs_times[-1])


def simulated_bayesian_locust(
    init_state,
    start_time,
    logging_times,
    base_model=LocustDynamics,
) -> State[torch.Tensor]:
    locust_model = bayesian_locust(base_model)
    with TorchDiffEq(), LogTrajectory(logging_times, is_traced=True) as lt:
        simulate(locust_model, init_state, start_time, logging_times[-1])
    return lt.trajectory



def get_locust_posterior_samples(
    guide, num_samples, init_state, start_time, logging_times
):
    locust_predictive = Predictive(
        simulated_bayesian_locust,
        guide,
        num_samples,
        init_state,
        start_time,
        logging_times,
    )
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
        ax=ax,
    )


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
    intervention=None,
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


def plot_ds_estimates(
    prior_samples,
    posterior_samples,
    group,
    which_coeff,
    ground_truth=False,
    true_attraction=None,
    true_wander=None,
    coef_names=None,
    xlim=0.5,
):
    
    if coef_names is None:
        coef_names = {
            "wander": ["w_sides", "w_inside", "w_outside", "w_feed"],
            "attraction": [
                "a_r", "a_l", "a_edge", "a_search", "a_feed"
            ],
        }

    fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    i = which_coeff

    sns.histplot(
        prior_samples[group][:, i], label="prior distribution", ax=ax[0], kde=False
    )

    ax[0].axvline(
        prior_samples[group][:, i].mean(),
        color="red",
        label="mean estimate",
        linestyle="--",
    )

    if ground_truth:
        if group == "attraction":
            ax[0].axvline(
                true_attraction[i], color="black", label="ground truth", linestyle="--"
            )
        if group == "wander":
            ax[0].axvline(true_wander[i], color="black", linestyle="--")

    ax[0].set_title(
        f"Prior ({coef_names[group][i]}, mean {round(prior_samples[group][:, i].mean().item(),3)})"
    )
    sns.despine(ax=ax[0])
    ax[0].set_yticks([])
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel(group)
    ax[0].set_xlim([0, xlim])

    sns.histplot(
        posterior_samples[group][:, i],
        label="posterior distribution",
        ax=ax[1],
        kde=False,
    )

    ax[1].axvline(
        posterior_samples[group][:, i].mean(),
        color="red",
        label="mean estimate",
        linestyle="--",
    )

    if ground_truth:
        if group == "attraction":
            ax[1].axvline(
                true_attraction[i], color="black", label="ground truth", linestyle="--"
            )
        if group == "wander":
            ax[1].axvline(true_wander[i], color="black", linestyle="--")

    ax[1].set_title(
        f"Posterior ({coef_names[group][i]}, mean {round(posterior_samples[group][:, i].mean().item(),3)})"
    )
    sns.despine(ax=ax[1])
    ax[1].set_yticks([])
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel(group)
    ax[1].set_xlim([0, xlim])

    plt.tight_layout()
    plt.show()


def plot_ds_interaction(posterior_samples, group, which_coeff, xlim=10, num_lines=20):
    coef_names = {
        "attraction": [
            "a_r", "a_l", "a_edge", "a_search", "a_feed"
        ],
    }

    i = which_coeff
    x = torch.arange(1, xlim, 1)
    posterior_samples = posterior_samples[group][:, i]

    ys = [x * posterior_samples[k] for k in range(num_lines)]

    for y in ys:
        sns.lineplot(x=x, y=y, color="grey", alpha=0.2, linewidth=0.5)

    plt.xlabel("units at target")
    plt.ylabel("proportion of units at at origin")
    plt.title(f"Contribution of {coef_names[group][which_coeff]} to flux term")
    sns.despine()
    plt.show()
