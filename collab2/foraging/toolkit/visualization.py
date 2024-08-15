from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictor(
    foragers: List[pd.DataFrame],
    predictor: List[List[pd.DataFrame]],
    predictorID: str,
    forager_index: List[int],
    time: List[int],
    grid_size: int,
    random_state: int = 0,
    size_multiplier: float = 10,
):
    """
    A function to visualize a computed predictor for specified foragers and timeframes.

    Parameters:
        - foragers : list of DataFrames containing forager positions, grouped by forager index
        - predictor : Nested list of DataFrames containing computed predictor scores, grouped by forager index and time
        - predictorID : Name of column containing predictor scores in `predictor`
        - forager_index : Index of foragers whose predictors are to be plotted
        - time : Timeframes for which predictor scores are to be plotted
        - grid_size : size of grid used to compute forager positions (used for setting x,y limits in figure)
        - random_state : used to choose plot colors for each forager
        - size_multiplier : used to select marker size in scatter plot

    Returns :
        NA
    """
    ncols = 4
    nrows = np.ceil(len(time) / ncols).astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    np.random.seed(random_state)
    random_colors = np.random.randint(0, 256, size=(len(forager_index), 3))
    # Convert the RGB values to hex format
    colors = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in random_colors]

    for i, t in enumerate(time):
        if isinstance(axes, np.ndarray):
            if axes.ndim == 2:
                r = i // ncols
                c = i % ncols
                ax = axes[r, c]
            else:
                ax = axes[i]
        else:
            ax = axes

        for j, f in enumerate(forager_index):
            if predictor[f][t] is not None:
                # normalize predictor value to choose scatter size and alpha
                size = predictor[f][t][predictorID] / predictor[f][t][predictorID].max()
                size[np.isnan(size)] = 0 
                ax.scatter(
                    predictor[f][t]["x"],
                    predictor[f][t]["y"],
                    s=size * size_multiplier,
                    color=colors[j],
                    alpha=size,
                )
            ax.scatter(
                foragers[f].loc[t, "x"],
                foragers[f].loc[t, "y"],
                s=50,
                marker="s",
                edgecolors="k",
                facecolors=colors[j],
            )

        ax.set_xlim([-1, grid_size])
        ax.set_ylim([-1, grid_size])
        ax.set_title(f"t={t}")
        ax.set_aspect("equal")

    # remove unused axes
    if len(time) % ncols and isinstance(axes, np.ndarray):
        if axes.ndim == 2:
            for c in range(len(time) % ncols, ncols):
                fig.delaxes(axes[nrows - 1, c])
        else:
            for c in range(len(time) % ncols, ncols):
                fig.delaxes(axes[c])

    fig.tight_layout(pad=2)
