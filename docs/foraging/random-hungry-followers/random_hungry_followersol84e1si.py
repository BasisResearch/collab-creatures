# %%NBQA-CELL-SEP967117
# importing packages. See https://github.com/BasisResearch/collab-creatures for repo setup
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean

from collab.utils import find_repo_root

root = find_repo_root()
from collab.foraging import random_hungry_followers as rhf
from collab.foraging import toolkit as ft

logging.basicConfig(format="%(message)s", level=logging.INFO)

# users can ignore smoke_test -- it's for automatic testing on GitHub, to make sure the notebook runs on future updates to the repository
smoke_test = "CI" in os.environ
num_frames = 5 if smoke_test else 50
num_svi_iters = 10 if smoke_test else 1000
num_samples = 10 if smoke_test else 1000


notebook_starts = time.time()

# %%NBQA-CELL-SEP967117
random.seed(23)
np.random.seed(23)

random_foragers_sim = rhf.RandomForagers(
    grid_size=40,
    probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
    num_foragers=3,
    num_frames=num_frames,
    num_rewards=15,
    grab_range=3,
)

# run a particular simulation with these parameters
random_foragers_sim()

# the results of the simulation are stored in `random_foragers_sim.foragersDF`.
# each row contains the x and y coordinates of a forager at a particular time

random_foragers_sim.foragersDF.head()

# %%NBQA-CELL-SEP967117
# You can visualize the foragers' movements using the `animate_foragers` function.
ft.animate_foragers(
    random_foragers_sim, plot_rewards=True, width=600, height=600, point_size=6
)

# %%NBQA-CELL-SEP967117
# add derived predictors to the simulation object

preferred_proximity = 4  # the distance at which foragers prefer to be from each other
random_foragers_derived = ft.derive_predictors(
    random_foragers_sim, optimal=preferred_proximity, dropna=False
)

# random_foragers_derived.derivedDF.to_csv('foragers1.csv', index=False)

# %%NBQA-CELL-SEP967117
# now we can plot the derived food traces in time:

ft.animate_foragers(
    random_foragers_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_traces=True,
)

# %%NBQA-CELL-SEP967117
# this exports and crops frame 0
if not smoke_test:
    trace_anim = ft.animate_foragers(
        random_foragers_derived,
        plot_rewards=True,
        width=600,
        height=600,
        point_size=10,
        plot_traces=True,
        produce_object=True,
    )

    trace_frame3 = go.Figure(trace_anim.frames[0].data, trace_anim.layout)
    trace_frame3.update_layout(title_text="food traces")

# this exports figures
# pio.write_image(
#     trace_frame3,
#     os.path.join(root, "docs/figures/fig2_trace.png"),
#     engine="kaleido",
#     width=600,
#     height=600,
#     scale=5,
# )

# trace_frame3_image = Image.open(os.path.join(root, "docs/figures/fig2_trace.png"))
# trace_frame3_cropped = trace_frame3_image.crop((600, 250, 3050, 2000))
# trace_frame3_cropped.save(os.path.join(root, "docs/figures/fig2_trace_cropped.png"))

# %%NBQA-CELL-SEP967117
# Each forager has a limited visibility range.
# We can plot this for one forager at a time
# as multiple foragers' visibility is hard to see
# let's say, forager 2

ft.animate_foragers(
    random_foragers_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_visibility=2,
    plot_traces=True,
)

# %%NBQA-CELL-SEP967117
# Plot the derived proximity score
# forager 2 again

ft.animate_foragers(
    random_foragers_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_proximity=2,
    proximity_multiplier=25,
)

# %%NBQA-CELL-SEP967117
# this exports and crops frame 3
prox_anim = ft.animate_foragers(
    random_foragers_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_proximity=2,
    proximity_multiplier=25,
    produce_object=True,
)

prox_frame3 = go.Figure(prox_anim.frames[0].data, prox_anim.layout)
prox_frame3.update_layout(title_text="Proximity scores (forager 2)")

# this exports figures
# pio.write_image(
#     prox_frame3,
#     os.path.join(root, "docs/figures/fig2_proximity.png"),
#     engine="kaleido",
#     width=600,
#     height=600,
#     scale=5,
# )

# prox_frame3_image = Image.open(os.path.join(root, "docs/figures/fig2_proximity.png"))
# prox_frame3_cropped = prox_frame3_image.crop((600, 250, 3050, 2000))
# prox_frame3_cropped.save(os.path.join(root, "docs/figures/fig2_proximity_cropped.png"))

# %%NBQA-CELL-SEP967117


def prep_data_for_inference(sim_derived):
    df = sim_derived.derivedDF[
        [
            "proximity_standardized",
            "trace_standardized",
            "visibility",
            "how_far_squared_scaled",
        ]
    ]

    df.dropna(inplace=True)

    for column in df.columns:
        df[column] = ft.normalize(df[column])

    data = torch.tensor(df.values, dtype=torch.float32)

    proximity, trace, visibility, how_far_score = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
    )

    print(
        str(len(proximity))
        + " data points prepared for inference, dropped "
        + str(len(sim_derived.derivedDF) - len(proximity))
        + " rows with missing values."
    )
    return proximity, trace, visibility, how_far_score


# %%NBQA-CELL-SEP967117
proximity, trace, visibility, how_far_score = prep_data_for_inference(
    random_foragers_derived
)

# %%NBQA-CELL-SEP967117
# plot how_far_score as a function of food trace and proximity score
ft.visualise_forager_predictors(trace, proximity, how_far_score)

# %%NBQA-CELL-SEP967117
# define the probabilistic model using pyro (https://pyro.ai/)
# p, t, v, b are the coefficients
# for proximity, trace, visibility, and the intercept

# ps, ts, vs, bs are analogous coefficients,
# but they contribute to the variance,
# which is not assumed to remain fixed


def model_sigmavar(proximity, trace, visibility, how_far_score):
    p = pyro.sample("p", dist.Normal(0, 0.2))
    t = pyro.sample("t", dist.Normal(0, 0.2))
    v = pyro.sample("v", dist.Normal(0, 0.2))
    b = pyro.sample("b", dist.Normal(0.5, 0.3))

    ps = pyro.sample("ps", dist.Exponential(7))
    ts = pyro.sample("ts", dist.Exponential(7))
    vs = pyro.sample("vs", dist.Exponential(7))
    bs = pyro.sample("bs", dist.Exponential(7))

    sigma = pyro.deterministic(
        "sigma", bs + ps * proximity + ts * trace + vs * visibility
    )
    mean = pyro.deterministic("mean", b + p * proximity + t * trace + v * visibility)

    with pyro.plate("data", len(how_far_score)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=how_far_score)


pyro.render_model(
    model_sigmavar,
    model_args=(proximity, trace, visibility, how_far_score),
    render_distributions=True,
)

# %%NBQA-CELL-SEP967117
# helper functions for inference, showing results


def summary(samples, sites):
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


def get_samples(
    proximity,
    trace,
    visibility,
    how_far_score,
    model=model_sigmavar,
    num_svi_iters=num_svi_iters,
    num_samples=num_samples,
):
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    svi = SVI(model_sigmavar, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

    iterations = []
    losses = []

    logging.info(f"Starting SVI inference with {num_svi_iters} iterations.")
    start_time = time.time()
    pyro.clear_param_store()
    for i in range(num_svi_iters):
        elbo = svi.step(proximity, trace, visibility, how_far_score)
        iterations.append(i)
        losses.append(elbo)
        if i % 200 == 0:
            logging.info("Elbo loss: {}".format(elbo))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("SVI inference completed in %.2f seconds.", elapsed_time)

    if not smoke_test:
        fig = px.line(
            x=iterations, y=losses, title="ELBO loss", template="presentation"
        )
        labels = {"iterations": "iteration", "losses": "loss"}
        fig.update_xaxes(showgrid=False, title_text=labels["iterations"])
        fig.update_yaxes(showgrid=False, title_text=labels["losses"])
        fig.update_layout(width=700)
        fig.show()

    predictive = Predictive(
        model, guide=guide, num_samples=num_samples, return_sites=["t", "p", "v"]
    )
    rhf_svi = {
        k: v.flatten().reshape(num_samples, -1).detach().cpu().numpy()
        for k, v in predictive(proximity, trace, visibility, how_far_score).items()
        if k != "obs"
    }

    print("SVI-based coefficient marginals:")
    for site, values in summary(rhf_svi, ["t", "p", "v"]).items():
        print("Site: {}".format(site))
        print(values, "\n")

    return {"svi_samples": rhf_svi, "svi_guide": guide, "svi_predictive": predictive}


def calculate_R_squared(guide):
    predictive = pyro.infer.Predictive(model_sigmavar, guide=guide, num_samples=1000)
    predictions = predictive(proximity, trace, visibility, how_far_score)

    simulated_outcome = (
        predictions["b"]
        + predictions["p"] * proximity
        + predictions["t"] * trace
        + predictions["v"] * visibility
    )

    mean_sim_outcome = simulated_outcome.mean(0).detach().cpu().numpy()

    observed_mean = torch.mean(how_far_score)

    tss = torch.sum((how_far_score - observed_mean) ** 2)
    rss = torch.sum((how_far_score - mean_sim_outcome) ** 2)

    r_squared = 1 - (rss / tss)

    return r_squared.float().item()


# %%NBQA-CELL-SEP967117
samples_random = get_samples(proximity, trace, visibility, how_far_score)

# %%NBQA-CELL-SEP967117
# plot the inferred posterior distributions of the coefficients
# Note that the model infers that the random foragers have
# near-zero coefficients for food trace and proximity score.
# The non-zero coefficient for visibility is consistent with the
# fact that the random foragers are mostly likely to move to nearby
# locations (higher visibility score) compared to farther locations.
ft.plot_coefs(
    samples_random, "Random foragers", nbins=120, ann_start_y=160, ann_break_y=50
)

# %%NBQA-CELL-SEP967117
print("R squared:", calculate_R_squared(samples_random["svi_guide"]))

# %%NBQA-CELL-SEP967117
random.seed(23)
np.random.seed(23)

hungry_sim = rhf.Foragers(
    grid_size=60, num_foragers=3, num_frames=num_frames, num_rewards=60, grab_range=3
)

hungry_sim()

hungry_sim = rhf.add_hungry_foragers(
    hungry_sim, num_hungry_foragers=3, rewards_decay=0.3, visibility_range=6
)

# %%NBQA-CELL-SEP967117
ft.animate_foragers(
    hungry_sim,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_traces=False,
)

# %%NBQA-CELL-SEP967117
# adding derived predictors
hungry_sim_derived = ft.derive_predictors(hungry_sim, dropna=False)

# if export is needed
# hungry_sim_derived.derivedDF.to_csv("foragers2.csv", index=False)

# plot with trace
# bird 2 again
ft.animate_foragers(
    hungry_sim_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_traces=True,
    trace_multiplier=7,
)

# %%NBQA-CELL-SEP967117
# now plot proximity
ft.animate_foragers(
    hungry_sim_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_proximity=1,
)

# %%NBQA-CELL-SEP967117
proximity, trace, visibility, how_far_score = prep_data_for_inference(
    hungry_sim_derived
)

# %%NBQA-CELL-SEP967117
# plot how_far_score as a function of proximity and food trace scores
# notice how all the points with high food trace scores
# have how_far_score close to 1,
# while how_far_score does not seem to systematically
# vary with proximity score

ft.visualise_forager_predictors(trace, proximity, how_far_score)

# %%NBQA-CELL-SEP967117
# run inference (SVI), and collect samples from the posterior
samples_hungry = get_samples(proximity, trace, visibility, how_far_score)

# %%NBQA-CELL-SEP967117
# plot the sampled posterior distributions of the coefficients
# Note that t, the coefficient for the food trace score, is high for hungry agents
ft.plot_coefs(
    samples_hungry, "Hungry foragers", nbins=160, ann_start_y=160, ann_break_y=10
)

# %%NBQA-CELL-SEP967117
print("R squared:", calculate_R_squared(samples_hungry["svi_guide"]))

# %%NBQA-CELL-SEP967117
random.seed(23)
np.random.seed(23)


follower_sim = rhf.Foragers(
    grid_size=60, num_foragers=3, num_frames=num_frames, num_rewards=30, grab_range=3
)
follower_sim()

follower_sim = rhf.add_follower_foragers(
    follower_sim,
    num_follower_foragers=3,
    visibility_range=6,
    getting_worse=0.5,
    optimal=3,
)

follower_sim_derived = ft.derive_predictors(
    follower_sim, getting_worse=0.5, optimal=3, visibility_range=6, dropna=False
)

# if export is needed
# follower_sim_derived.derivedDF.to_csv("foragers3.csv", index=False)

# %%NBQA-CELL-SEP967117
ft.animate_foragers(
    follower_sim_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_proximity=2,
)

# %%NBQA-CELL-SEP967117
proximity, trace, visibility, how_far_score = prep_data_for_inference(
    follower_sim_derived
)

# %%NBQA-CELL-SEP967117
# plot how_far_score as a function of the derived predictor scores (food trace and proximity score)
# Observe that the how_far_score is high when the proximity score is high, but doesn't appear to vary systematically with the food trace score
ft.visualise_forager_predictors(trace, proximity, how_far_score)

# %%NBQA-CELL-SEP967117
# run inference (SVI), and collect samples from the posterior
samples_followers = get_samples(proximity, trace, visibility, how_far_score)

# %%NBQA-CELL-SEP967117
# plot the sampled posterior distributions of the coefficients
# Note that p, the coefficient for the proximity score, is high for follower agents
ft.plot_coefs(
    samples_followers, "Follower foragers", nbins=140, ann_start_y=160, ann_break_y=10
)

# %%NBQA-CELL-SEP967117
print("R squared:", calculate_R_squared(samples_followers["svi_guide"]))

# %%NBQA-CELL-SEP967117


def plot_all_coefficients(
    svi_samples, groups, title, xrange, yrange, generate_object=False
):
    data = {"p": [], "t": [], "inference_method": [], "group": []}

    for i, svi_sample in enumerate(svi_samples):
        site_p_svi = svi_sample["p"].flatten().tolist()
        site_t_svi = svi_sample["t"].flatten().tolist()
        data["p"] += site_p_svi
        data["t"] += site_t_svi
        data["inference_method"] += ["svi"] * len(site_p_svi)
        data["group"] += [groups[i]] * len(site_p_svi)

    df = pd.DataFrame(data)

    fig = px.scatter(
        df, x="p", y="t", color="group", title=title, template="presentation"
    )

    fig.update_xaxes(range=[xrange[0], xrange[1]], showgrid=False)
    fig.update_yaxes(range=[yrange[0], yrange[1]], showgrid=False)

    fig.update_layout(autosize=False, width=600, height=600)

    fig.update_traces(marker=dict(size=12, opacity=0.1))

    if generate_object:
        return fig
    else:
        fig.show()


# %%NBQA-CELL-SEP967117
# plot the inferred coefficients for all three bird types in one scatter plot
# Note that the coefficients for the food trace score (t) are high for hungry agents
# while the coefficients for the proximity score (p) are high for follower agents
# and the random agents have near-zero coefficients for both predictors
svi_samples = [
    samples_random["svi_samples"],
    samples_hungry["svi_samples"],
    samples_followers["svi_samples"],
]

groups = ["random", "hungry", "follower"]

fig2_rhf = plot_all_coefficients(
    svi_samples,
    groups,
    "Coefficients for three bird types",
    [-0.2, 0.5],
    [-1, 2.6],
    generate_object=True,
)

fig2_rhf.update_layout(legend=dict(yanchor="top", y=1.05, xanchor="right", x=0.97))

fig2_rhf.show()

# %%NBQA-CELL-SEP967117
# this exports fig2_rfh

# pio.write_image(
#     fig2_rhf,
#     os.path.join(root, "docs/figures/fig2_rfh.png"),
#     engine="kaleido",
#     width=600,
#     height=600,
#     scale=5,
# )

# %%NBQA-CELL-SEP967117
notebook_ends = time.time()
print(
    "notebook took",
    notebook_ends - notebook_starts,
    "seconds, that is ",
    (notebook_ends - notebook_starts) / 60,
    "minutes to run",
)
