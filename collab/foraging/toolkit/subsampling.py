import random

import numpy as np


def subset_frames_evenly_spaced(df_raw, desired_frames=300):
    df = df_raw.copy()
    print("original_frames:", df["time"].max())
    num_frames = df["time"].max()
    print("original_shape:", df.shape)
    df["time"] = np.round(df["time"] / (num_frames / desired_frames)).astype(int) + 1
    df = df.drop_duplicates(subset=["time", "forager"], keep="first")
    df = df[df["time"] <= desired_frames]
    print("resulting_frames:", df["time"].max())
    print("resulting_shape:", df.shape)

    print("min_time", df["time"].min())
    print("max_time", df["time"].max())

    return df


# updated function to allow user to pass min and max
def rescale_to_grid(df, size, gridMin=None, gridMax=None):
    def rescale_column(column, size=size, gridMin=None, gridMax=None):
        if gridMin is None:
            gridMin = column.min()

        if gridMax is None:
            gridMax = column.max()

        mapped = (column - gridMin) / (gridMax - gridMin)
        rescaled = np.floor(mapped * (size - 1)) + 1
        return rescaled

    df["x"] = rescale_column(df["x"], size, gridMin, gridMax)
    df["y"] = rescale_column(df["y"], size, gridMin, gridMax)
    return df


def sample_time_slices(df, proportion):
    """Samples a proportion of the timeframes from a dataframe."""
    n_frames = max(df["time"])
    n_sample = int(n_frames * proportion)
    return df[df["time"].isin(random.sample(range(n_frames), n_sample))]
