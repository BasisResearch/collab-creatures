# %%NBQA-CELL-SEPf56012
# importing packages. See https://github.com/BasisResearch/collab-creatures for repo setup
import os
import pickle
import time

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch

from collab.foraging import locust as lc
from collab.foraging import toolkit as ft
from collab.utils import find_repo_root

root = find_repo_root()

# users can ignore smoke_test -- it's for automatic testing on GitHub, to make sure the notebook runs on future updates to the repository
smoke_test = "CI" in os.environ
subset_starts = 420
subset_ends = 430 if smoke_test else 480
desired_frames = 500 if smoke_test else 900
num_iterations = 50 if smoke_test else 2000
num_samples = 20 if smoke_test else 1000
sample_size = 100 if smoke_test else 1000
locust_sample_size = 100 if smoke_test else 178770

# developer_mode runs inference; if False, it loads saved version of the results
developer_mode = False
if developer_mode:
    smoke_test = False

notebook_starts = time.time()


# %%NBQA-CELL-SEPf56012
# Load the data

locust_data_path = os.path.join(root, "data/foraging/locust/15EQ20191202_tracked.csv")

df = lc.load_and_clean_locust(
    path=locust_data_path,
    desired_frames=desired_frames,
    grid_size=45,
    rewards_x=[0.68074, -0.69292],
    rewards_y=[-0.03068, -0.03068],
    subset_starts=subset_starts,
    subset_ends=subset_ends,
)

loc_subset = df["subset"]
loc_all = df["all_frames"]


# %%NBQA-CELL-SEPf56012
# uncomment this if you're interested in processing the whole dataset
# instead of the subsample

# start_time = time.time()
# loc_all = ft.derive_predictors(loc_all, sampling_rate= .1)
# end_time = time.time()
# print("time taken", end_time - start_time)
# takes about 5 minutes to process the whole dataset
# with sampling rate .1


# %%NBQA-CELL-SEPf56012
# plot histogram of distances between foragers
locust_distances = ft.foragers_to_forager_distances(loc_subset)
ft.distances_and_peaks(locust_distances)


# %%NBQA-CELL-SEPf56012
# this illustrates the proximity function that we will use
# in predictor derivation

start = 0.0
end = 5.0
step = 0.001
x = [start + i * step for i in range(int((end - start) / step) + 1)]
y = [
    ft.proximity_score(_d, getting_worse=0.0001, optimal=2.11, proximity_decay=3)
    for _d in x
]
plt.plot(x, y)
plt.title("Proximity score used in predictor derivation")
plt.xlabel("Distance")
plt.ylabel("Proximity score")
plt.show()


# %%NBQA-CELL-SEPf56012
loc_subset = ft.derive_predictors(
    loc_subset,
    rewards_decay=0.4,
    visibility_range=90,
    getting_worse=0.001,
    optimal=2.11,
    proximity_decay=3,
    generate_communicates_indicator=True,
    info_time_decay=10,
    info_spatial_decay=0.1,
    finders_tolerance=2,
    time_shift=subset_starts - 1,
    sampling_rate=0.1,
    restrict_to_invisible=False,
)


# %%NBQA-CELL-SEPf56012
# animate the communication trace.
ft.animate_foragers(
    loc_subset,
    plot_rewards=True,
    width=600,
    height=600,
    point_size=10,
    plot_communicate=1,
    plot_traces=True,
    trace_multiplier=15,
    communicate_multiplier=1,
)


# %%NBQA-CELL-SEPf56012
# bin the data for robust inference
loc_subset_robust = ft.prep_data_for_robust_inference(
    loc_subset, gridsize=9
)  # modifies loc_subset in place as well


# %%NBQA-CELL-SEPf56012
data = ft.get_tensorized_data(loc_subset)

proximity, trace, visibility, communicate, how_far = (
    data["proximity_standardized"],
    data["trace_standardized"],
    data["visibility"],
    data["communicate_standardized"],
    data["how_far"],
)

# todo refactor
# ft.visualise_forager_predictors(trace, proximity, how_far, com=communicate)


# %%NBQA-CELL-SEPf56012
# plot how_far_score as a function of food trace and proximity score
ft.visualise_forager_predictors(trace, proximity, how_far)

# Or, plot heatmaps to better see densities of points.
# todo build into the toolkit
x_to_plot = [trace, proximity, communicate]
x_labels = ["trace", "proximity", "communicate"]
y = how_far

# if seaborn gives deprecation warnings, here's a hacky fix
# import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)

Colors = ["lightblue", "black"]
cmap = colors.LinearSegmentedColormap.from_list("lightBlueToBlack", Colors)

for idx, x in enumerate(x_to_plot):
    # Create the jointplot
    g = sns.jointplot(
        x=x,
        y=y,
        kind="hex",
        gridsize=50,
        cmap=cmap,
        norm=colors.LogNorm(),
        joint_kws=dict(facecolor="k"),
        marginal_kws=dict(facecolor="lightblue", edgecolor="black"),
    )
    g.set_axis_labels(xlabel=x_labels[idx], ylabel="how_far_score")

    plt.show()


# %%NBQA-CELL-SEPf56012
# format variables
locust = loc_subset.derivedDF

locust["proximity_id"] = locust.proximity_cat.astype("category").cat.codes
locust["trace_id"] = locust.trace_cat.astype("category").cat.codes
locust["communicate_id"] = locust.communicate_cat.astype("category").cat.codes
locust["how_far"] = locust.how_far_squared_scaled


# %%NBQA-CELL-SEPf56012
# subsample the data if doing automatic testing
locust_sample_size = len(locust) if not smoke_test else 10
locust_subsample = locust.sample(n=locust_sample_size, random_state=42)


locust_tensorized = {}

locust_tensorized["how_far"] = torch.tensor(
    locust_subsample["how_far"].values, dtype=torch.float32
)
locust_tensorized["proximity"] = torch.tensor(
    locust_subsample["proximity_standardized"].values, dtype=torch.float32
)
locust_tensorized["trace"] = torch.tensor(
    locust_subsample["trace_standardized"].values, dtype=torch.float32
)
locust_tensorized["communicate"] = torch.tensor(
    locust_subsample["communicate_standardized"].values, dtype=torch.float32
)

# potentially useful for robust inference, not needed for now
# missing_categories = {}
# def list_empty_categories(column):
#     return list(column.cat.categories[column.value_counts().eq(0)].values)


# columns_to_clean_categories = ["proximity_id", "trace_id", "communicate_id"]
# for column in columns_to_clean_categories:
#     locust_subsample[column] = locust_subsample.proximity_cat.astype("category")
#     missing_categories[column] = list_empty_categories(locust_subsample[column])
#     locust_subsample[column] = locust_subsample[column].cat.remove_unused_categories()
#     locust_subsample[column] = locust_subsample[column].cat.codes.values

#     locust_tensorized[column] = torch.tensor(
#         locust_subsample[column].values, dtype=torch.int32
#     )


# %%NBQA-CELL-SEPf56012
# define the model for inference
how_far_tensor = locust_tensorized["how_far"]
proximity_tensor = locust_tensorized["proximity"]
communicate_tensor = locust_tensorized["communicate"]
trace_tensor = locust_tensorized["trace"]
# not using visibility as locust see everything in the area


def continuous_model(
    proximity_tensor, trace_tensor, communicate_tensor, how_far_tensor
):
    bias = pyro.sample("bias", dist.Normal(0.5, 0.3))
    p = pyro.sample("p", dist.Normal(0.0, 0.2))
    t = pyro.sample("t", dist.Normal(0.0, 0.2))
    c = pyro.sample("c", dist.Normal(0.0, 0.2))

    sigma_bias = pyro.sample("sigma_bias", dist.Exponential(7))
    p_sigma = pyro.sample("p_sigma", dist.Exponential(7))
    t_sigma = pyro.sample("t_sigma", dist.Exponential(7))
    c_sigma = pyro.sample("c_sigma", dist.Exponential(7))

    with pyro.plate("data", len(how_far_tensor)):
        sigma = pyro.deterministic(
            "sigma",
            sigma_bias
            + p_sigma * proximity_tensor
            + t_sigma * trace_tensor
            + c_sigma * communicate_tensor,
        )

        mean = pyro.deterministic(
            "mean",
            bias + p * proximity_tensor + t * trace_tensor + c * communicate_tensor,
        )

        how_far_observed = pyro.sample(
            "how_far_observed", dist.Normal(mean, sigma), obs=how_far_tensor
        )


# %%NBQA-CELL-SEPf56012
# run inference (SVI) if in developer mode
if developer_mode:
    pyro.clear_param_store

    guide_continuous = pyro.infer.autoguide.AutoDiagonalNormal(continuous_model)
    Adam = pyro.optim.Adam
    Trace_ELBO = pyro.infer.Trace_ELBO
    SVI = pyro.infer.SVI

    pyro.clear_param_store()
    svi = SVI(
        model=continuous_model,
        guide=guide_continuous,
        optim=Adam({"lr": 0.01}),
        loss=Trace_ELBO(),
    )

    losses = []
    num_steps = num_iterations
    for step in range(num_steps):
        loss = svi.step(
            proximity_tensor, trace_tensor, communicate_tensor, how_far_tensor
        )
        losses.append(loss)
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss}")

    plt.plot(losses)


# %%NBQA-CELL-SEPf56012
# sample from the posterior if in developer mode
if developer_mode:
    locust_data_folder = os.path.join(root, "data/foraging/locust/")
    locust_continuous_samples_path = os.path.join(
        locust_data_folder,
        f"locust_cont__samples_iter{num_iterations}_lsamples{locust_sample_size}.pkl",
    )

    sites = ["bias", "p", "t", "c"]

    num_samples = num_samples
    predictive_continuous = pyro.infer.Predictive(
        model=continuous_model,
        guide=guide_continuous,
        num_samples=num_samples,
        return_sites=sites,
    )

    samples_continuous = predictive_continuous(
        proximity_tensor, trace_tensor, communicate_tensor, how_far_tensor
    )

    with open(locust_continuous_samples_path, "wb") as f:
        pickle.dump(samples_continuous, f)


# %%NBQA-CELL-SEPf56012
# plot the posterior distribution of the coefficients
if not developer_mode:
    locust_data_folder = os.path.join(root, "data/foraging/locust/")
    locust_continuous_samples_path = os.path.join(
        locust_data_folder,
        f"locust_cont__samples_iter2000_lsamples178770.pkl",
    )


with open(locust_continuous_samples_path, "rb") as f:
    samples_continuous = pickle.load(f)

svi_samples = samples_continuous.copy()
svi_samples.pop("bias")

ft.plot_coefs(
    svi_samples,
    title="Locust model coefficients",
    nbins=80,
    ann_start_y=160,
    ann_break_y=10,
)


# %%NBQA-CELL-SEPf56012
# note that while the impact of communication is detected
# the model as a whole is not amazing at predicting how far
# this is not surprising as linear regression is a rough approximation here

simulated_outcome = (
    samples_continuous["bias"]
    + samples_continuous["p"] * proximity_tensor
    + samples_continuous["t"] * trace_tensor
    + samples_continuous["c"] * communicate_tensor
)

mean_sim_outcome = simulated_outcome.mean(0).detach().cpu().numpy()
observed_mean = torch.mean(how_far_tensor)
tss = torch.sum((how_far_tensor - observed_mean) ** 2)
rss = torch.sum((how_far_tensor - mean_sim_outcome) ** 2)
r_squared = 1 - (rss / tss)

print("r squared is:", r_squared.float().item())

notebook_ends = time.time()
print(
    "notebook took",
    notebook_ends - notebook_starts,
    "seconds, that is ",
    (notebook_ends - notebook_starts) / 60,
    "minutes to run",
)
