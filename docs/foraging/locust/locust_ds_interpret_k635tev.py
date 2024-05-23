# %%NBQA-CELL-SEP967117
import os

import dill
import matplotlib.pyplot as plt
import pyro
import seaborn as sns

pyro.settings.set(module_local_params=True)

sns.set_style("white")

seed = 123
pyro.clear_param_store()
pyro.set_rng_seed(seed)

import matplotlib.pyplot as plt
import seaborn as sns

from collab.foraging import locust as lc
from collab.utils import find_repo_root

# %%NBQA-CELL-SEP967117
locds = lc.LocustDS(
    data_code="15EQ20191202",
    start=0,
    end=10,
)

locds.run_inference(
    "length",
    num_iterations=1500,
    num_samples=150,
    lr=0.001,  # force=True, save=True
)

# %%NBQA-CELL-SEP967117
locds.get_prior_samples(num_samples=150)
locds.plot_param_estimates(3, 4)

# %%NBQA-CELL-SEP967117
lc.plot_ds_interaction(locds.samples, "attraction", 4, xlim=10, num_lines=100)

# %%NBQA-CELL-SEP967117
data_code = "15EQ20191202"

samples_a_feed_10 = []

root = find_repo_root()
data_path = os.path.join(root, f"data/foraging/locust/ds/locust_samples_a_feed_10.pkl")

if os.path.exists(data_path):
    with open(data_path, "rb") as f:
        samples_a_feed_10 = dill.load(f)
else:
    for start in [0, 30, 60, 90, 120, 150]:
        print(start)
        locds = lc.LocustDS(
            data_code=data_code,
            start=start,
            end=start + 10,
        )

        locds.run_inference("length", num_iterations=1500, num_samples=150, lr=0.001)
        samples_a_feed_10.append(locds.samples["attraction"][:, 4])

    with open(data_path, "wb") as f:
        dill.dump(samples_a_feed_10, f)

# %%NBQA-CELL-SEP967117
start = [0, 30, 60, 90, 120, 150]

fig, axs = plt.subplots(len(start), 1, figsize=(8, 6), sharex=True)

for i, (s, tensor) in enumerate(zip(start, samples_a_feed_10)):
    axs[i].hist(tensor, bins=30, alpha=0.5, label=f"start at {s}")
    axs[i].set_ylabel("frequency")
    axs[i].title.set_text(f"start at {s}")
    axs[i].set_xlim(0, 0.3)

axs[-1].set_xlabel("$a_{feed}$")
plt.tight_layout()
# plt.savefig("locust_a_feed_10_various_starts.png")
plt.show()
