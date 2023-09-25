import numpy as np
import pandas as pd

from collab import foraging_toolkit as ft


# from utils import generate_grid


# adding trace of rewards
def rewards_trace(distance, rewards_decay):
    return np.exp(-rewards_decay * distance)


def rewards_to_trace(
    rewards,
    grid_size,
    num_frames,
    rewards_decay=0.5,
    start=None,
    end=None,
    time_shift=0,
    grid=None,
):
    if start is None:
        start = 0

    if end is None:
        end = num_frames

    if grid is None:
        grid = ft.generate_grid(grid_size)

    traces = []

    for t in range(start, end):
        rewt = rewards[t]
        trace = grid.copy()
        trace["trace"] = 0
        trace["time"] = t + 1
        trace["trace_standardized"] = 0

        if len(rewt) > 0:
            for re in range(len(rewt)):
                trace["trace"] += rewards_trace(
                    np.sqrt((rewt["x"].iloc[re] - trace["x"]) ** 2 + (rewt["y"].iloc[re] - trace["y"]) ** 2),
                    rewards_decay,
                )

            trace["trace_standardized"] = (trace["trace"] - trace["trace"].mean()) / trace["trace"].std()

            trace["time"] = trace["time"] + time_shift

        traces.append(trace)

    tracesDF = pd.concat(traces)

    return {"traces": traces, "tracesDF": tracesDF}
