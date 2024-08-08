from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictor(
    foragers: List[pd.DataFrame],
    predictor: List[List[pd.DataFrame]],
    predictorID: str,
    f: List[int],
    t: List[int],
    grid_size: int,
    random_state: int = 0,
    size_multiplier: float = 10,
):
    ncols = 4
    nrows = np.ceil(len(t) / ncols).astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    np.random.seed(random_state)
    random_colors = np.random.randint(0, 256, size=(len(f), 3))
    # Convert the RGB values to hex format
    colors = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in random_colors]

    for i, ti in enumerate(t):
        if isinstance(axes, np.ndarray):
            if axes.ndim == 2:
                r = i // ncols
                c = i % ncols
                ax = axes[r, c]
            else:
                ax = axes[i]
        else:
            ax = axes

        for j, fj in enumerate(f):
            if predictor[fj][ti] is not None:
                # normalize predictor value to choose scatter size
                size = (
                    predictor[fj][ti][predictorID]
                    / predictor[fj][ti][predictorID].max()
                )
                ax.scatter(
                    predictor[fj][ti]["x"],
                    predictor[fj][ti]["y"],
                    s=0.2 + size * size_multiplier,
                    color=colors[j],
                    alpha=0.4,
                )
            ax.scatter(
                foragers[fj].loc[ti, "x"],
                foragers[fj].loc[ti, "y"],
                s=50,
                marker="s",
                edgecolors="k",
                facecolors=colors[j],
            )

        ax.set_xlim([-1, grid_size])
        ax.set_ylim([-1, grid_size])
        ax.set_title(f"t={ti}")
        ax.set_aspect("equal")

    # remove unused axes
    if len(t) % ncols and isinstance(axes, np.ndarray):
        if axes.ndim == 2:
            for c in range(len(t) % ncols, ncols):
                fig.delaxes(axes[nrows - 1, c])
        else:
            for c in range(len(t) % ncols, ncols):
                fig.delaxes(axes[c])

    fig.tight_layout(pad=2)
