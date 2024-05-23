# %%NBQA-CELL-SEPf56012
import os
import time

import pyro
import seaborn as sns
import torch

pyro.settings.set(module_local_params=True)

sns.set_style("white")

seed = 123
pyro.clear_param_store()
pyro.set_rng_seed(seed)

import seaborn as sns
import torch

from collab.foraging import locust as lc

smoke_test = "CI" in os.environ
start = 30
end = start + 2 if smoke_test else start + 12
num_samples = 10 if smoke_test else 150
num_lines = 5 if smoke_test else 100
num_iterations = 10 if smoke_test else 1500
notebook_starts = time.time()


# %%NBQA-CELL-SEPf56012
# this instantiates the dynamics,
# the data will be used in calibration later on

locds = lc.LocustDS(
    data_code="15EQ20191202",
    start=start,
    end=end,
)


# %%NBQA-CELL-SEPf56012
# before we calibrate
# we already can simulate forward by passing some
# parameters of our choice

# a_r, a_l, a_edge, a_search, a_feed
true_attraction = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])

# w_sides, w_inside, w_outside, w_feed
true_wander = torch.tensor([0.2, 0.1, 0.01, 0.05])

# by default we take the initial state from the data
# the user can pass `init_state=...` argument
locds.simulate_trajectories(
    true_attraction=true_attraction,
    true_wander=true_wander,
)


locds.plot_simulated_trajectories()


# %%NBQA-CELL-SEPf56012
locds.get_prior_samples(num_samples=num_samples, force=False)

# note, if you run with your own data, use force=True.
# Here we don't use force as we use pre-comupted samples
# obtained in the training and validation for multiple models
# which takes a long time to run.

# we plot prior samples later, when we contrast
# them with the posterior samples after training


# %%NBQA-CELL-SEPf56012
# once you get prior samples
# locds.prior_samples are available for downstream use
# for instance, to plot the prior predictions for a compartment
# using `plot_multiple_trajectories`.

locds.plot_multiple_trajectories("feed_l", num_lines=num_lines, priors=True)


# %%NBQA-CELL-SEPf56012
locds.run_inference(
    "length",
    num_iterations=num_iterations,
    num_samples=num_samples,
    lr=0.0005,
    # force=True,
    save=True,
)

# if you inspect the convergence by setting force=True, save = False,
# notice loss drops but the range of sites visited is still fairly wide
# this is because small changes in params can lead to large changes in
# a dynamical system, so loss even with low lr is a bit jumpy


# %%NBQA-CELL-SEPf56012
# once you run inference, the posterior samples are available as
# locds.samples, also available for downstream use, such as plotting

locds.plot_multiple_trajectories("feed_l", num_lines=num_lines, priors=False)


# %%NBQA-CELL-SEPf56012
# now we compare the date (black, dashed) to
# mean predictions obtained from the posterior samples,
# the shaded area is the 95% credible interval of the mean
# and does not represent the variation in observations themselves
# which can be inspected using `.plot_multiple_trajectories()`

locds.posterior_check()


# %%NBQA-CELL-SEPf56012
locds.evaluate(samples=locds.samples, subset=locds.subset)


# %%NBQA-CELL-SEPf56012
locds.plot_param_estimates(3, 3, xlim=0.35)


# %%NBQA-CELL-SEPf56012
locds.validate(
    validation_data_code="15EQ20191205",
    num_iterations=num_iterations,
    num_samples=num_samples,
)  # , force = True, save = True)


# %%NBQA-CELL-SEPf56012
locds.posterior_check(
    samples=locds.v_samples, subset=locds.v_subset, save=False
)  # this also might a png file in docs/figures if save=True


# %%NBQA-CELL-SEPf56012
notebook_ends = time.time()
print(
    "notebook took",
    notebook_ends - notebook_starts,
    "seconds, that is ",
    (notebook_ends - notebook_starts) / 60,
    "minutes to run",
)
