from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictor(
    foragers: List[pd.DataFrame],
    predictor: List[List[pd.DataFrame]],
    predictor_name: str,
    forager_position_indices: List[int],
    forager_predictor_indices: List[int],
    time: List[int],
    grid_size: int,
    random_state: Optional[int] = 0,
    size_multiplier: Optional[int] = 10,
):
    """
    A function to visualize a computed predictor for specified foragers and timeframes.

    Parameters:
        - foragers : list of DataFrames containing forager positions, grouped by forager index
        - predictor : Nested list of DataFrames containing computed predictor scores, grouped by forager index and time
        - predictor_name : Name of column containing predictor scores in `predictor`
        - forager_position_indices: List of indices of foragers whose positions are to be plotted
        - forager_predictor_indices : List of indices of foragers whose predictors are to be plotted
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
    random_colors = np.random.randint(0, 256, size=(len(forager_position_indices), 3))
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

        # plot forager positions
        for j, f in enumerate(forager_position_indices):
            ax.scatter(
                foragers[f].loc[t, "x"],
                foragers[f].loc[t, "y"],
                s=50,
                marker="s",
                edgecolors="k",
                facecolors=colors[j],
            )

        # plot predictor values
        for j, f in enumerate(forager_predictor_indices):
            if predictor[f][t] is not None:
                # normalize predictor value to choose scatter size and alpha
                size = (
                    abs(predictor[f][t][predictor_name])
                    / abs(predictor[f][t][predictor_name]).max()
                )
                size[np.isnan(size)] = 0
                ax.scatter(
                    predictor[f][t]["x"],
                    predictor[f][t]["y"],
                    s=size * size_multiplier,
                    color=colors[j],
                    alpha=abs(size * 0.8),
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

    fig.suptitle(f"Predictor: {predictor_name}")
    fig.tight_layout(pad=2)
    fig.show()


def animate_predictors(
    foragersDF: pd.DataFrame,
    predictor: List[List[pd.DataFrame]],
    predictor_name: str,
    forager_position_indices: List[int],
    forager_predictor_indices: List[int],
    grid_size: int,
    random_state: Optional[int] = 0,
    size_multiplier: Optional[int] = 10,
):
    """
    A function to animate a computed predictor for specified foragers.

    Parameters:
        - foragersDF : flattened DataFrame of forager positions
        - predictor : Nested list of DataFrames containing computed predictor scores, grouped by forager index and time
        - predictor_name : Name of column containing predictor scores in `predictor`
        - forager_position_incices : Index of foragers whose positions are to be plotted
        - forager_predictor_indices : Index of foragers whose predictors are to be plotted
        - grid_size : size of grid used to compute forager positions (used for setting x,y limits in figure)
        - random_state : used to choose plot colors for each forager
        - size_multiplier : used to select marker size in scatter plot

    Returns :
        - ani : animation
    """
    trajectory_data = foragersDF[foragersDF["forager"].isin(forager_position_indices)]

    num_foragers = foragersDF["forager"].nunique()
    num_frames = foragersDF["time"].nunique()

    # Generate random colors for each particle
    np.random.seed(random_state)
    colors = np.random.rand(num_foragers, 3)  # RGB values for face colors

    # Create a figure and axis
    fig, ax = plt.subplots()
    foragers_scat = ax.scatter([], [], s=50, marker="s", facecolor=[], edgecolor=[])
    predictors_scat_list = []
    for i in range(len(forager_predictor_indices)):
        predictors_scat_list.append(
            ax.scatter([], [], s=[], facecolor=[], edgecolor=[])
        )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    # TODO potentially expand with forager legend
    # ax.legend()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Initialize function to set up the background of each frame
    def init():
        foragers_scat.set_offsets(np.empty((0, 2)))
        foragers_scat.set_facecolor(np.array([]))  # Set face color array
        foragers_scat.set_edgecolor(np.array([]))  # Set edge color array

        for predictors_scat in predictors_scat_list:
            predictors_scat.set_offsets(np.empty((0, 2)))
            predictors_scat.set_facecolor(np.array([]))  # Set face color array
            predictors_scat.set_edgecolor(np.array([]))  # Set edge color array
            predictors_scat.set_sizes(np.array([]))

        return foragers_scat, *predictors_scat_list

    # Update function for each frame
    def update(frame):

        # Update positions of the particles
        current_positions = trajectory_data.loc[
            trajectory_data["time"] == frame, ["x", "y"]
        ].values
        foragers_scat.set_offsets(current_positions)

        # Update face and edge colors of the particles
        foragers_scat.set_facecolor(
            colors[forager_position_indices]
        )  # Set face colors directly
        foragers_scat.set_edgecolor(
            colors[forager_position_indices]
        )  # Set edge colors directly

        # Update predictor
        for i, f in enumerate(forager_predictor_indices):
            if predictor[f][frame] is not None:
                current_features = predictor[f][frame].loc[:, ["x", "y"]]
                size = (
                    abs(predictor[f][frame][predictor_name])
                    / abs(predictor[f][frame][predictor_name]).max()
                )
                size[np.isnan(size)] = 0
                predictors_scat_list[i].set_offsets(current_features)
                predictors_scat_list[i].set_sizes(size * size_multiplier)
                predictors_scat_list[i].set_alpha(size)
                predictors_scat_list[i].set_facecolor(colors[f])
                predictors_scat_list[i].set_edgecolor(colors[f])
            else:
                predictors_scat_list[i].set_offsets(np.empty((0, 2)))

        return foragers_scat, *predictors_scat_list

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        init_func=init,
        blit=True,
        interval=500,
        repeat_delay=3500,
    )
    return ani
