import random
from typing import Optional

import numpy as np
import pandas as pd


# this function does not give a consistent frame-rate because of rounding.
def subset_frames_evenly_spaced(
    df_raw: pd.DataFrame, desired_frames: int = 300
) -> pd.DataFrame:
    df = df_raw.copy()
    # start time at 0
    df["time"] = df["time"] - df["time"].min()
    num_frames = df["time"].max() + 1
    print("original_frames:", num_frames)
    print("original_shape:", df.shape)
    df["time"] = np.floor(df["time"] / (num_frames - 1) * (desired_frames - 1)).astype(
        int
    )
    # df["time"] = np.ceil(df["time"] / (num_frames / (desired_frames-1))).astype(int)
    df = df.drop_duplicates(subset=["time", "forager"], keep="first").reset_index(
        drop=True
    )
    print("resulting_frames:", df["time"].nunique())
    print("resulting_shape:", df.shape)
    return df


# another version of subsampling that can be used for cases when frame-rate is important (eg. velocity)
def subsample_frames_constant_frame_rate(
    df_raw: pd.DataFrame, frame_spacing: int = 2, fps: Optional[float] = None
) -> pd.DataFrame:
    df = df_raw.copy()
    # start time at 0
    df["time"] = df["time"] - df["time"].min()
    og_frames = df["time"].max() + 1
    print("original_frames:", og_frames)
    print("original_shape:", df.shape)

    keep_ind = df["time"] % frame_spacing == 0
    df = df.loc[keep_ind].reset_index(drop=True)
    df["time"] = (df["time"] / frame_spacing).astype(int)
    new_frames = df["time"].nunique()
    print("resulting_frames:", new_frames)
    print("resulting_shape:", df.shape)
    if fps is not None:
        print(f"new frame-rate = {fps*new_frames/og_frames : .2f}")
    return df


# updated function to allow user to pass gridMin and gridMax
# grid points start at 0
def rescale_to_grid(
    df_raw: pd.DataFrame,
    size: int,
    gridMin: Optional[float] = None,
    gridMax: Optional[float] = None,
):
    def rescale_column(
        column: pd.DataFrame,
        size: int,
        gridMin: Optional[float] = None,
        gridMax: Optional[float] = None,
    ):
        if gridMin is None:
            gridMin = column.min()

        if gridMax is None:
            gridMax = column.max()

        mapped = (column - gridMin) / (gridMax - gridMin)
        rescaled = np.floor(mapped * size)
        rescaled[rescaled > size - 1] = size - 1
        rescaled[rescaled < 0] = 0

        return rescaled

    df = df_raw.copy()
    df["x"] = rescale_column(df["x"], size, gridMin, gridMax)
    df["y"] = rescale_column(df["y"], size, gridMin, gridMax)
    return df


def sample_time_slices(df, proportion):
    """Samples a proportion of the timeframes from a dataframe."""
    n_frames = max(df["time"])
    n_sample = int(n_frames * proportion)
    return df[df["time"].isin(random.sample(range(n_frames), n_sample))]
