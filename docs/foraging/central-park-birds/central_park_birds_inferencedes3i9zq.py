# %%NBQA-CELL-SEPf56012
import logging
import os
import time

import dill
import numpy as np
import pandas as pd
import plotly.io as pio
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import torch
from plotly import express as px
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean

from collab.foraging import toolkit as ft
from collab.utils import find_repo_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
root = find_repo_root()

smoke_test = "CI" in os.environ
num_svi_iters = 50 if smoke_test else 1000
num_samples = 50 if smoke_test else 1000
keys = [50] if smoke_test else [10, 20, 30, 40, 50, 60, 70, 80]
sampling_rate = 0.01 if smoke_test else 0.01

notebook_starts = time.time()


# %%NBQA-CELL-SEPf56012
# this file is generated using `central_park_birds_predictors.ipynb`
path = os.path.join(
    root,
    f"data/foraging/central_park_birds_cleaned_2022/central_park_objects_sampling_rate_{sampling_rate}.pkl",
)

if not smoke_test:
    assert os.path.exists(
        path
    ), "Please run `central_park_birds_predictors.ipynb` to prep the data first."

    with open(path, "rb") as file:
        central_park_objects = dill.load(file)


# %%NBQA-CELL-SEPf56012
def cp_prep_data_for_iference(obj):
    df = obj.how_farDF.copy()
    print("Initial dataset size:", len(df))
    df.dropna(inplace=True)
    print("After dropping NAs:", len(df))

    columns_to_normalize = [
        "distance",
        "proximity_standardized",
    ]

    for column in columns_to_normalize:
        df[column] = ft.normalize(df[column])

    return (
        torch.tensor(df["distance"].values),
        torch.tensor(df["proximity_standardized"].values),
        torch.tensor(df["how_far_squared_scaled"].values),
    )


# %%NBQA-CELL-SEPf56012
def model_sigmavar_proximity(distance, proximity, how_far):
    d = pyro.sample("d", dist.Normal(0, 0.2))
    p = pyro.sample("p", dist.Normal(0, 0.2))
    b = pyro.sample("b", dist.Normal(0.5, 0.3))

    ds = pyro.sample("ds", dist.Exponential(7))
    ps = pyro.sample("ps", dist.Exponential(7))
    bs = pyro.sample("bs", dist.Exponential(7))

    sigma = bs + ds * distance + ps * proximity
    mean = b + d * distance + p * proximity

    with pyro.plate("data", len(how_far)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=how_far)


# %%NBQA-CELL-SEPf56012
def get_samples(
    distance,
    proximity,
    how_far,
    model=model_sigmavar_proximity,
    num_svi_iters=num_svi_iters,
    num_samples=num_samples,
):
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    svi = SVI(
        model_sigmavar_proximity, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO()
    )

    iterations = []
    losses = []

    logging.info(f"Starting SVI inference with {num_svi_iters} iterations.")
    start_time = time.time()
    pyro.clear_param_store()
    for i in range(num_svi_iters):
        elbo = svi.step(distance, proximity, how_far)
        iterations.append(i)
        losses.append(elbo)
        if i % 50 == 0:
            logging.info("Elbo loss: {}".format(elbo))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("SVI inference completed in %.2f seconds.", elapsed_time)

    # uncomment if you want to see the ELBO loss plots
    # fig = px.line(x=iterations, y=losses, title="ELBO loss", template="presentation")
    # labels = {"iterations": "iteration", "losses": "loss"}
    # fig.update_xaxes(showgrid=False, title_text=labels["iterations"])
    # fig.update_yaxes(showgrid=False, title_text=labels["losses"])
    # fig.update_layout(width=700)
    # fig.show()

    predictive = Predictive(model, guide=guide, num_samples=num_samples)

    proximity_svi = {
        k: v.flatten().reshape(num_samples, -1).detach().cpu().numpy()
        for k, v in predictive(distance, proximity, how_far).items()
        if k != "obs"
    }

    print("SVI-based coefficient marginals:")
    for site, values in ft.summary(proximity_svi, ["d", "p"]).items():
        print("Site: {}".format(site))
        print(values, "\n")

    return {
        "svi_samples": proximity_svi,
        "svi_guide": guide,
        "svi_predictive": predictive,
    }


# %%NBQA-CELL-SEPf56012
path = os.path.join(
    root, "data/foraging/central_park_birds_cleaned_2022/duck_outcomes.pkl"
)

if os.path.exists(path):
    print("The duck samples exist, skipping inference, will load later on.")

if not smoke_test and not os.path.exists(path):
    ducks_objects = central_park_objects[0]
    # for ducks starting that low might not make sense
    # [19, 46, 85]
    duck_outcomes = {}

    for key in keys:
        obj = ducks_objects[key]
        print(f"Working on ducks with optimal={key}")
        distance, proximity, how_far = cp_prep_data_for_iference(obj)
        ft.visualise_forager_predictors(
            distance,
            proximity,
            how_far,
            vis_sampling_rate=0.05,
            titles=[f"Distance (ducks)", f"Proximity (ducks, optimal={key})"],
            x_axis_labels=["distance", "proximity"],
        )
        duck_outcomes[key] = get_samples(distance, proximity, how_far)

        # helps prevent crashing the kernel on slower machines
        time.sleep(1)

    with open(path, "wb") as file:
        dill.dump(duck_outcomes, file)


# %%NBQA-CELL-SEPf56012
def plot_coefs(outcomes, title, ann_start_y=100, ann_break_y=50, generate_object=False):
    keys = [10, 20, 30, 40, 50, 60, 70, 80]
    # [19, 46, 85]
    samples = {}

    for key in keys:
        samples[key] = outcomes[key]["svi_samples"]["p"].flatten()

    samples_df = pd.DataFrame(samples)
    # samples_df_medians = samples_df.median(axis=0).tolist()

    fig_coefs = px.histogram(
        samples_df,
        template="presentation",
        opacity=0.4,
        labels={"variable": "preferred proximity", "value": "proximity coefficient"},
        width=700,
        title=title,
    )

    # for i, color in enumerate(['#1f77b4', '#ff7f0e', '#2ca02c']):
    #         fig_coefs.add_vline(x=samples_df_medians[i], line_dash="dash", line_color=color, name=f"Median ({samples_df_medians[i]})")

    #         fig_coefs.add_annotation(
    #         x=samples_df_medians[i],
    #         y= ann_start_y + ann_break_y * i,  # Adjust the vertical position of the label
    #         text=f"{samples_df_medians[i]:.2f}",
    #         bgcolor="white",
    #         showarrow=False,
    #         opacity=0.8,
    #         )

    fig_coefs.update_layout(
        barmode="overlay"
    )  # , yaxis=dict(showticklabels=False, title=None, showgrid=False))

    if generate_object:
        return fig_coefs
    else:
        fig_coefs.show()


# %%NBQA-CELL-SEPf56012
duck_outcomes_path = os.path.join(
    root, "data/foraging/central_park_birds_cleaned_2022/duck_outcomes.pkl"
)

if not smoke_test:
    duck_outcomes = dill.load(open(duck_outcomes_path, "rb"))

    ducks_coefs_plot = plot_coefs(
        duck_outcomes, "Ducks", ann_start_y=350, ann_break_y=50, generate_object=True
    )

    ducks_coefs_plot.show()

    pio.write_image(
        ducks_coefs_plot,
        os.path.join(root, "docs/figures/duck_coefs_plot.png"),
        engine="kaleido",
        width=700,
        scale=5,
    )


# %%NBQA-CELL-SEPf56012
def calculate_R_squared_prox(distance, proximity, how_far, guide, subsample_size=1000):
    predictive = pyro.infer.Predictive(
        model_sigmavar_proximity, guide=guide, num_samples=1000
    )

    random_indices = np.random.choice(len(distance), size=subsample_size, replace=False)
    distance_sub = distance[random_indices]
    proximity_sub = proximity[random_indices]
    how_far_sub = how_far[random_indices]

    predictions = predictive(distance_sub, proximity_sub, how_far_sub)

    simulated_outcome = (
        predictions["b"] + predictions["p"] * proximity + predictions["d"] * distance
    )

    mean_sim_outcome = simulated_outcome.mean(0).detach().cpu().numpy()

    observed_mean = torch.mean(how_far)

    tss = torch.sum((how_far - observed_mean) ** 2)
    rss = torch.sum((how_far - mean_sim_outcome) ** 2)

    r_squared = 1 - (rss / tss)

    return r_squared.float().item()


# %%NBQA-CELL-SEPf56012
if not smoke_test:
    ducks_objects = central_park_objects[0]

    for key in keys:
        distance, proximity, how_far = cp_prep_data_for_iference(ducks_objects[key])
        guide = duck_outcomes[key]["svi_guide"]
        print(
            f"R^2 for ducks with optimal={key}:",
            calculate_R_squared_prox(distance, proximity, how_far, guide),
        )

# interestingly, knowing where they won't go is useful


# %%NBQA-CELL-SEPf56012
path = os.path.join(
    root, "data/foraging/central_park_birds_cleaned_2022/sps_outcomes.pkl"
)

if not smoke_test and not os.path.exists(path):
    sps_objects = central_park_objects[1]  # [19, 46, 85]

    sps_outcomes = {}

    for key in keys:
        obj = sps_objects[key]
        print(f"Working on sparrows et al. with optimal={key}")
        distance, proximity, how_far = cp_prep_data_for_iference(obj)
        ft.visualise_forager_predictors(
            distance,
            proximity,
            how_far,
            vis_sampling_rate=0.05,
            titles=[
                f"Distance (sparrows et al.)",
                f"Proximity (sparrows et al., optimal={key})",
            ],
            x_axis_labels=["distance", "proximity"],
        )

        sps_outcomes[key] = get_samples(distance, proximity, how_far)

        time.sleep(1)

    with open(path, "wb") as file:
        dill.dump(sps_outcomes, file)


# %%NBQA-CELL-SEPf56012
sps_outcomes_path = os.path.join(
    root, "data/foraging/central_park_birds_cleaned_2022/sps_outcomes.pkl"
)

if not smoke_test:
    sps_outcomes = dill.load(open(sps_outcomes_path, "rb"))

    sps_coefs_plot = plot_coefs(
        sps_outcomes,
        "Sparrows et al.",
        ann_start_y=200,
        ann_break_y=30,
        generate_object=True,
    )

    sps_coefs_plot.show()
    # add title to figure

    pio.write_image(
        sps_coefs_plot,
        os.path.join(root, "docs/figures/sps_coefs_plot.png"),
        engine="kaleido",
        width=700,
        scale=5,
    )


# %%NBQA-CELL-SEPf56012
# note sparrows' movements are harder to predict

if not smoke_test:
    sps_objects = central_park_objects[1]

    for key in keys:
        distance, proximity, how_far = cp_prep_data_for_iference(sps_objects[key])
        guide = sps_outcomes[key]["svi_guide"]
        print(
            f"R^2 for sparrows et al. with optimal={key}:",
            calculate_R_squared_prox(distance, proximity, how_far, guide),
        )

notebook_ends = time.time()

print(
    f"notebook took {notebook_ends - notebook_starts} seconds, {(notebook_ends - notebook_starts)/60} minutes to run"
)
