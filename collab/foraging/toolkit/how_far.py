import numpy as np
import pandas as pd


def add_how_far_squared_scaled(sim):
    foragers = sim.foragers
    how_far = sim.visibility.copy()

    for b in range(sim.num_foragers):
        for t in range(sim.num_frames - 2):
            try:
                x_new = int(foragers[b]["x"][t + 1])
                y_new = int(foragers[b]["y"][t + 1])
            except (KeyError, AttributeError):
                x_new = int(foragers[b]["x"].iloc[t + 1])
                y_new = int(foragers[b]["y"].iloc[t + 1])

            # print(how_far[b][t].head(n=1))
            _hf = how_far[b][t]
            _hf["how_far_squared"] = (_hf["x"] - x_new) ** 2 + (_hf["y"] - y_new) ** 2
            _hf["how_far_squared_scaled"] = (
                -_hf["how_far_squared"]
                / (2 * (sim.step_size_max + sim.visibility_range) ** 2)
                + 1
            )

        if len(how_far[b]) > (sim.num_frames - 1):
            how_far[b][sim.num_frames - 1]["how_far_squared"] = np.nan
            how_far[b][sim.num_frames - 1]["how_far_squared_scaled"] = np.nan

    sim.how_far = how_far

    foragers_how_far = []
    for forager in range(sim.num_foragers):
        foragers_how_far.append(pd.concat(how_far[forager]))

    sim.how_farDF = pd.concat(foragers_how_far)
