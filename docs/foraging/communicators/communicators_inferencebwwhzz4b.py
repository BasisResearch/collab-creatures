# %%NBQA-CELL-SEPf56012
import logging
import os
import time

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

root = find_repo_root()

smoke_test = "CI" in os.environ

num_frames = 5 if smoke_test else 50
num_svi_iters = 10 if smoke_test else 1000
num_samples = 10 if smoke_test else 1000

notebook_starts = time.time()


# %%NBQA-CELL-SEPf56012
sim_params = pd.read_csv(
    os.path.join(
        root, "data/foraging/communicators/communicators_strong/metadataDF.csv"
    ),
    index_col=0,
)
home_dir = os.path.join(root, "data/foraging/communicators/communicators_strong")

sim0_folder = "sim0_run0"
sim6_folder = "sim1_run0"

sim0_dir = os.path.join(home_dir, sim0_folder)
sim6_dir = os.path.join(home_dir, sim6_folder)

bird0 = pd.read_csv(os.path.join(sim0_dir, "foragerlocsDF.csv"), index_col=0)

foragerlocsDF0 = pd.read_csv(os.path.join(sim0_dir, "foragerlocsDF.csv"), index_col=0)
foragerlocsDF6 = pd.read_csv(os.path.join(sim6_dir, "foragerlocsDF.csv"), index_col=0)

rewardlocsDF0 = pd.read_csv(os.path.join(sim0_dir, "rewardlocsDF.csv"), index_col=0)
rewardlocsDF6 = pd.read_csv(os.path.join(sim6_dir, "rewardlocsDF.csv"), index_col=0)


# drop last frame to make the two dataframes the same length
last = foragerlocsDF0["time"].unique()[-1]
foragerlocsDF0 = foragerlocsDF0.drop(
    foragerlocsDF0[foragerlocsDF0["time"] == last].index
)
foragerlocsDF6 = foragerlocsDF6.drop(
    foragerlocsDF6[foragerlocsDF6["time"] == last].index
)

assert all(rewardlocsDF0["time"].unique() == foragerlocsDF0["time"].unique())
assert all(rewardlocsDF6["time"].unique() == foragerlocsDF6["time"].unique())

communicators0 = ft.object_from_data(
    foragersDF=foragerlocsDF0, rewardsDF=rewardlocsDF0, calculate_step_size_max=True
)
communicators6 = ft.object_from_data(
    foragersDF=foragerlocsDF6, rewardsDF=rewardlocsDF6, calculate_step_size_max=True
)


# %%NBQA-CELL-SEPf56012
# first, take a look at birs that don't communicate
if not smoke_test:
    ft.animate_foragers(
        communicators0, plot_rewards=True, width=600, height=600, point_size=10
    )


# %%NBQA-CELL-SEPf56012
# now compare this with communicators with communication coefficient .6
if not smoke_test:
    ft.animate_foragers(
        communicators6, plot_rewards=True, width=600, height=600, point_size=10
    )


# %%NBQA-CELL-SEPf56012
communicators0_derived = ft.derive_predictors(
    communicators0, generate_communicates_indicator=True, dropna=False
)


# %%NBQA-CELL-SEPf56012
communicators6_derived = ft.derive_predictors(
    communicators6,
    generate_communicates_indicator=True,
    dropna=False,
)


# %%NBQA-CELL-SEPf56012
ft.animate_foragers(
    communicators0_derived,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_visibility=0,
    plot_communicate=4,
    plot_traces=True,
    communicate_multiplier=0.5,
    trace_multiplier=1,
)


# %%NBQA-CELL-SEPf56012
def prep_data_for__communicators_inference(sim_derived):
    print("Initial dataset size:", sim_derived.derivedDF.shape[0])
    df = sim_derived.derivedDF.copy().dropna()

    for column in [
        "trace_standardized",
        "proximity_standardized",
        "communicate_standardized",
    ]:
        df[column] = ft.normalize(df[column])

    df.dropna(inplace=True)

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


# %%NBQA-CELL-SEPf56012
(
    trace0,
    proximity0,
    visibility0,
    communicate0,
    how_far_score0,
) = prep_data_for__communicators_inference(communicators0_derived)

(
    trace6,
    proximity6,
    visibility6,
    communicate6,
    how_far_score6,
) = prep_data_for__communicators_inference(communicators6_derived)


# %%NBQA-CELL-SEPf56012
ft.visualise_forager_predictors(trace0, proximity0, how_far_score0, com=communicate0)


# %%NBQA-CELL-SEPf56012
ft.visualise_forager_predictors(trace6, proximity6, how_far_score6, com=communicate6)


# %%NBQA-CELL-SEPf56012
def model_sigmavar_com(proximity, trace, visibility, communicate, how_far_score):
    p = pyro.sample("p", dist.Normal(0, 0.2))
    t = pyro.sample("t", dist.Normal(0, 0.2))
    v = pyro.sample("v", dist.Normal(0, 0.2))
    c = pyro.sample("c", dist.Normal(0, 0.2))
    b = pyro.sample("b", dist.Normal(0.5, 0.3))

    ps = pyro.sample("ps", dist.Exponential(7))
    ts = pyro.sample("ts", dist.Exponential(7))
    vs = pyro.sample("vs", dist.Exponential(7))
    cs = pyro.sample("cs", dist.Exponential(7))
    bs = pyro.sample("bs", dist.Exponential(7))

    sigma = pyro.deterministic(
        "sigma", bs + ps * proximity + ts * trace + vs * visibility + cs * communicate
    )
    mean = pyro.deterministic(
        "mean", b + p * proximity + t * trace + v * visibility + c * communicate
    )

    with pyro.plate("data", len(how_far_score)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=how_far_score)


# %%NBQA-CELL-SEPf56012
def get_samples(
    proximity,
    trace,
    visibility,
    communicate,
    how_far_score,
    model=model_sigmavar_com,
    num_svi_iters=num_svi_iters,
    num_samples=num_samples,
):
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    svi = SVI(model_sigmavar_com, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

    iterations = []
    losses = []

    logging.info(f"Starting SVI inference with {num_svi_iters} iterations.")
    start_time = time.time()
    pyro.clear_param_store()
    for i in range(num_svi_iters):
        elbo = svi.step(proximity, trace, visibility, communicate, how_far_score)
        iterations.append(i)
        losses.append(elbo)
        if i % 200 == 0:
            logging.info("Elbo loss: {}".format(elbo))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("SVI inference completed in %.2f seconds.", elapsed_time)

    fig = px.line(x=iterations, y=losses, title="ELBO loss", template="presentation")
    labels = {"iterations": "iteration", "losses": "loss"}
    fig.update_xaxes(showgrid=False, title_text=labels["iterations"])
    fig.update_yaxes(showgrid=False, title_text=labels["losses"])
    fig.update_layout(width=700)
    fig.show()

    predictive = Predictive(
        model, guide=guide, num_samples=num_samples, return_sites=["t", "p", "c"]
    )
    communicate_svi = {
        k: v.flatten().reshape(num_samples, -1).detach().cpu().numpy()
        for k, v in predictive(
            proximity, trace, visibility, communicate, how_far_score
        ).items()
        if k != "obs"
    }

    print("SVI-based coefficient marginals:")
    for site, values in ft.summary(communicate_svi, ["t", "p", "c"]).items():
        print("Site: {}".format(site))
        print(values, "\n")

    return {
        "svi_samples": communicate_svi,
        "svi_guide": guide,
        "svi_predictive": predictive,
    }


# %%NBQA-CELL-SEPf56012
samples0 = get_samples(proximity0, trace0, visibility0, communicate0, how_far_score0)


# %%NBQA-CELL-SEPf56012
samples6 = get_samples(proximity6, trace6, visibility6, communicate6, how_far_score6)


# %%NBQA-CELL-SEPf56012
def calculate_R_squared_com(
    proximity, trace, visibility, communicate, how_far_score, guide
):
    predictive = pyro.infer.Predictive(
        model_sigmavar_com, guide=guide, num_samples=1000
    )
    predictions = predictive(proximity, trace, visibility, communicate, how_far_score)

    simulated_outcome = (
        predictions["b"]
        + predictions["p"] * proximity
        + predictions["t"] * trace
        + predictions["v"] * visibility
        + predictions["c"] * communicate
    )

    mean_sim_outcome = simulated_outcome.mean(0).detach().cpu().numpy()

    observed_mean = torch.mean(how_far_score)

    tss = torch.sum((how_far_score - observed_mean) ** 2)
    rss = torch.sum((how_far_score - mean_sim_outcome) ** 2)

    r_squared = 1 - (rss / tss)

    return r_squared.float().item()


# %%NBQA-CELL-SEPf56012
if not smoke_test:
    print(
        calculate_R_squared_com(
            proximity0,
            trace0,
            visibility0,
            communicate0,
            how_far_score0,
            samples0["svi_guide"],
        )
    )

    print(
        calculate_R_squared_com(
            proximity6,
            trace6,
            visibility6,
            communicate6,
            how_far_score6,
            samples6["svi_guide"],
        )
    )


# %%NBQA-CELL-SEPf56012
def plot_coefs(samples, title, ann_start_y=100, ann_break_y=50, generate_object=False):
    for key in samples["svi_samples"].keys():
        samples["svi_samples"][key] = samples["svi_samples"][key].flatten()

    samplesDF = pd.DataFrame(samples["svi_samples"])
    samplesDF_medians = samplesDF.median(axis=0)

    fig_coefs = px.histogram(
        samplesDF,
        template="presentation",
        opacity=0.4,
        labels={"variable": "coefficient"},
        width=700,
        title=title,
        marginal="rug",
    )

    for i, color in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c"]):
        fig_coefs.add_vline(
            x=samplesDF_medians[i],
            line_dash="dash",
            line_color=color,
            name=f"Median ({samplesDF_medians[i]})",
        )

        fig_coefs.add_annotation(
            x=samplesDF_medians[i],
            y=ann_start_y
            + ann_break_y * i,  # Adjust the vertical position of the label
            text=f"{samplesDF_medians[i]:.2f}",
            showarrow=False,
            bordercolor="black",
            borderwidth=2,
            bgcolor="white",
            opacity=0.8,
        )

    fig_coefs.update_traces(
        marker=dict(line=dict(width=2, color="Black"))
    )  # Add black outline to bars

    if generate_object:
        return fig_coefs
    else:
        fig_coefs.show()


# %%NBQA-CELL-SEPf56012
plot0 = ft.plot_coefs(
    samples0,
    "No communication",
    nbins=100,
    ann_start_y=100,
    ann_break_y=30,
    generate_object=True,
)

if not smoke_test:
    pio.write_image(
        plot0,
        os.path.join(root, "docs/figures/fig3_no_communication.png"),
        engine="kaleido",
        width=600,
        height=600,
        scale=5,
    )


# %%NBQA-CELL-SEPf56012
plot6 = ft.plot_coefs(
    samples6,
    "Communication coefficient = 0.6",
    nbins=100,
    ann_start_y=100,
    ann_break_y=30,
    generate_object=True,
)

plot6.show()

if not smoke_test:
    pio.write_image(
        plot6,
        os.path.join(root, "docs/figures/fig3_communication06.png"),
        engine="kaleido",
        width=600,
        height=600,
        scale=5,
    )


# %%NBQA-CELL-SEPf56012
notebook_ends = time.time()
print(
    "notebook took",
    notebook_ends - notebook_starts,
    "seconds, that is ",
    (notebook_ends - notebook_starts) / 60,
    "minutes to run",
)
