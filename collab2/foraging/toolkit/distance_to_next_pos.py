import numpy as np
import pandas as pd
import copy

def _generate_distance_to_next_pos(foragers, local_windows, window_size):
    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    distance_to_next_pos = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames - 1):
            if distance_to_next_pos[f][t] is not None:
                x_new = foragers[f].at[t+1,"x"]
                y_new = foragers[f].at[t+1,"y"]
                if np.isfinite(x_new) and np.isfinite(y_new):
                    distance_to_next_pos[f][t]["raw_distance_to_next_pos"] = np.sqrt((distance_to_next_pos[f][t]["x"] - x_new)**2 + (distance_to_next_pos[f][t]["y"] - y_new)**2)
                    distance_to_next_pos[f][t]["scored_distance_to_next_pos"] = 1 - (distance_to_next_pos[f][t]["raw_distance_to_next_pos"]/(2*window_size))**2
                    #normalization by rescaling
                    #distance_to_next_pos[f][t]["scored_distance_to_next_pos"] = distance_to_next_pos[f][t]["scored_distance_to_next_pos"]/(distance_to_next_pos[f][t]["scored_distance_to_next_pos"].abs().max())
                else:
                    distance_to_next_pos[f][t]["raw_distance_to_next_pos"] = np.nan
                    distance_to_next_pos[f][t]["scored_distance_to_next_pos"] = np.nan

        #save nans for last frame
        distance_to_next_pos[f][num_frames-1]["raw_distance_to_next_pos"] = np.nan
        distance_to_next_pos[f][num_frames-1]["scored_distance_to_next_pos"] = np.nan

    return distance_to_next_pos            