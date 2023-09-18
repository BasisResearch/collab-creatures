import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# cleaning functions
# remove big jumps
def remove_jumps(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, max_jump=10):
    """Removes jumps in the data that are larger than max_jump."""
    Y = np.copy(tracks.T)
    num_frames, num_body_parts, _, num_tracks = Y.shape
    new_instance_scores = []
    new_point_scores = []
    new_tracking_scores = []
    new_occupancy = []

    def diffs_to_next_non_nan(y):
        """Calculates the difference between each point and the next non-NaN point."""
        # first calculate the normal diffs
        diffs = np.diff(y)
        # then replace the diffs after a NaN with the diff to the next non-NaN
        nan_indices = np.where(np.isnan(diffs))[0]
        for idx in nan_indices:
            # check if there's a non-NaN value after the NaN
            if np.sum(~np.isnan(y[idx+1:])) > 0:
                # if there is, find the index of the next non-NaN value
                next_non_nan_idx = np.where(~np.isnan(y[idx+1:]))[0][0] + idx + 1
                # replace the NaN diff with the diff to the next non-NaN value
                diffs[idx] = y[next_non_nan_idx] - y[idx]
        return diffs

    for track in range(num_tracks):

        # calculate the distance between each point and the next non-NaN point
        diffs_x = diffs_to_next_non_nan(Y[:, 0, 0, track])
        diffs_y = diffs_to_next_non_nan(Y[:, 0, 1, track])
        diffs = np.sqrt(diffs_x**2 + diffs_y**2)
            
        # Identify where the jumps exceed the threshold
        jump_indices = np.where(diffs > max_jump)[0]
        track_updated = track # will keep adding tracks to Y... this is the index of the last track added
        for idx in jump_indices:
            # Create a new track with initial part of the original track
            new_track = np.copy(Y[:, :, :, track_updated])
            new_track[:(idx+1)] = np.nan# Y[idx+1:, :, :, track]
            
            # Set the original track values after the break point to NaN
            Y[(idx+1):, :, :, track_updated] = np.nan

            # append new track to Y
            Y = np.concatenate([Y, new_track[..., np.newaxis]], axis=3)
            track_updated = Y.shape[3] - 1

            # Keep track of other variables scores
            # new_tracks.append(new_track)
            new_instance_scores.append(instance_scores[track])
            new_point_scores.append(point_scores[track])
            new_tracking_scores.append(tracking_scores[track])
            new_occupancy.append((~np.isnan(new_track[:, 0, 0])).astype(int))
          
    print('removing jumps resulted in ' + str(np.shape(Y)[3]) + ' tracks')
    if new_instance_scores: # if track_names is not empty
        # update other variables
        # Y = np.concatenate([Y, np.array(new_tracks).transpose(1,2,3,0)], axis=3)
        instance_scores = np.vstack([instance_scores, new_instance_scores])
        point_scores = np.concatenate([point_scores, new_point_scores], axis=0)
        tracking_scores = np.vstack([tracking_scores, new_tracking_scores])

        track_occupancy = np.hstack([track_occupancy, np.array(new_occupancy).transpose()])
        track_names.extend([f'track_{num_tracks+i}' for i in range(np.shape(Y)[3]-num_tracks)])

    tracks = Y.T
    return tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores

def combine_all_adjacent_tracks(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, max_distance=10, max_time=10):
    num_tracks_old, num_body_parts, _, num_frames = tracks.shape
    max_iter = 100
    for i in range(max_iter):
        tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores = combine_next_adjacent_tracks(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, max_distance=max_distance, max_time=max_time)
        num_tracks_new, num_body_parts, _, num_frames = tracks.shape
        if num_tracks_new == num_tracks_old:
            break
        else:
            num_tracks_old = num_tracks_new
    return  tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores

# merge tracks that are close together in time and space
def combine_next_adjacent_tracks(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, max_distance=10, max_time=10):
    """Combines tracks that are close together in time and space."""
    Y = np.copy(tracks.T)

    num_frames, num_body_parts, _, num_tracks = Y.shape

    # We'll keep a list of tracks that should be removed after merging
    tracks_to_remove = []

    for track1 in range(num_tracks):
        if track1 in tracks_to_remove:
            continue
        if np.sum(~np.isnan(Y[:, 0, 0, track1])) == 0:
            continue
        end_of_track1 = np.where(~np.isnan(Y[:, 0, 0, track1]))[0][-1]
        pos_end_track1 = Y[end_of_track1, 0, :, track1]

        min_distance = float('inf')
        best_track2 = None
        for track2 in range(num_tracks): 
            if track1 == track2:
                continue
            if np.sum(~np.isnan(Y[:, 0, 0, track2])) == 0:
                continue
            if track2 in tracks_to_remove:
                continue
            start_of_track2 = np.where(~np.isnan(Y[:, 0, 0, track2]))[0][0]
            pos_start_track2 = Y[start_of_track2, 0, :, track2]

            time_diff = start_of_track2 - end_of_track1
            spatial_diff = np.sqrt((pos_end_track1[0] - pos_start_track2[0])**2 + (pos_end_track1[1] - pos_start_track2[1])**2)

            if 0 < time_diff <= max_time and spatial_diff < min_distance: 
                min_distance = spatial_diff
                best_track2 = track2

        if best_track2 is not None and min_distance <= max_distance:
            # Insert the second track into the first track 
            to_insert = np.copy(Y[:, :, :, best_track2])
            original = np.copy(Y[:, :, :, track1])
            # take whichever is non nan at each point
            Y[:, :, :, track1] = np.where(np.isnan(original), to_insert, original)
            # Y[start_of_track2:, :, :, best_track2] = np.nan
            tracks_to_remove.append(best_track2)
            
    # Remove the merged tracks
    tracks = np.delete(Y.T, tracks_to_remove, axis=0)
    instance_scores = np.delete(instance_scores, tracks_to_remove, axis=0)
    point_scores = np.delete(point_scores, tracks_to_remove, axis=0)
    track_names = np.delete(np.array(track_names), tracks_to_remove).tolist()
    track_occupancy = np.delete(track_occupancy, tracks_to_remove, axis=1)
    tracking_scores = np.delete(tracking_scores, tracks_to_remove, axis=0)
    print('merging in ' + str(len(tracks_to_remove)) + ' tracks')

    return tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores


# remove tracks that are too short (based on head location only)
def remove_short_tracks(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, min_length=10):
    """Removes tracks that are shorter than min_length."""
    Y = np.copy(tracks.T)
    num_frames, num_body_parts, _, num_tracks = Y.shape

    # Create a mask to identify tracks that need to be retained
    retain_mask = np.ones(num_tracks, dtype=bool)

    for track in range(num_tracks):
        # Identify where the track is not NaN
        track_indices = np.where(~np.isnan(Y[:, 0, 0, track]))[0]
        # If the track is shorter than the threshold, mark it for deletion
        if len(track_indices) < min_length:
            retain_mask[track] = False
    print('removing '+str(np.sum(~retain_mask))+ ' tracks, out of ' + str(num_tracks))
    # Use the retain_mask to filter out short tracks from all variables
    tracks = tracks[retain_mask]
    instance_scores = instance_scores[retain_mask]
    point_scores = point_scores[retain_mask]
    track_names = np.array(track_names)[retain_mask].tolist()
    track_occupancy = track_occupancy[:, retain_mask]
    tracking_scores = tracking_scores[retain_mask]
    return tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores




# interpolate missing values
def fill_missing(tracks, kind="linear"):
    """Fills missing values independently along each dimension after the first."""
    Y = np.copy(tracks.T)
    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        
        # If no valid data or only one data point, skip this iteration
        if np.flatnonzero(~np.isnan(y)).size <= 1:
            continue

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    tracks = Y.T
    print('interpolated missing points')
    return tracks

# smooth tracks
def smooth_tracks(tracks, smooth_win=25, smooth_poly=3):
    """Smooths tracks using a Savitzky-Golay filter."""
    # Ensure the window size is odd
    if smooth_win % 2 == 0:
        print('warning: window should be odd')
        smooth_win += 1
    
    Y = tracks.T
    num_frames, num_body_parts, _, num_tracks = Y.shape

    for track in range(num_tracks):
        for part in range(num_body_parts):
            for dim in range(2): # x and y
                valid_indices = ~np.isnan(Y[:, part, dim, track])
                # Only smooth if there are any valid values for the track
                if np.any(valid_indices):
                    valid_data = Y[valid_indices, part, dim, track]
                    smooth_win_use = np.min([smooth_win, len(valid_data)])
                    smooth_poly_use = np.min([smooth_poly, smooth_win_use-1, len(valid_data)-1])
                    # Apply the filter
                    Y[valid_indices, part, dim, track] = savgol_filter(valid_data, smooth_win_use, smooth_poly_use)
    print('smoothed tracks')
    return Y.T


def generate_cleaned_analysis_file(filename, max_jump = 50, merge_max_distance = 5, merge_max_time = 50, min_length = 10, smooth_win = 15, smooth_poly = 3):
    base, extension = filename.rsplit('.', 1)
    new_filename = f"{base}_cleaned.{extension}"

    # Open the original file in read mode and the new file in write mode
    # Open the original file in read mode
    with h5py.File(filename, 'r') as f, h5py.File(new_filename, 'w') as g:

        # Copy all datasets from the old file to the new file
        for name, item in f.items():
            f.copy(item, g, name)

        # Load data from the new file
        dset_names = list(g.keys())
        tracks = g["tracks"][:]
        instance_scores = f["instance_scores"][:]
        point_scores = f["point_scores"][:]
        track_names = [name.decode() for name in f["track_names"][:]]
        track_occupancy = f["track_occupancy"][:]
        tracking_scores = f["tracking_scores"][:]

        # # Clean data....
        print('Pre-cleaning, total of ' + str(len(track_names)) + ' tracks')
        # # remove big jumps
        tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores = remove_jumps(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, max_jump=max_jump)

        # interpolate missing values
        tracks = fill_missing(tracks)

        # merge tracks that are close together in time and space
        tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores = combine_all_adjacent_tracks(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, max_distance=merge_max_distance, max_time=merge_max_time)

        # interpolate missing values
        tracks = fill_missing(tracks)

        # remove tracks that are too short
        tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores = remove_short_tracks(tracks, instance_scores, point_scores, track_names, track_occupancy, tracking_scores, min_length=min_length)

        # smooth tracks
        tracks = smooth_tracks(tracks, smooth_win=smooth_win, smooth_poly=smooth_poly)
        print('Total of '+ str(len(track_names)) + ' tracks')
        # Delete old datasets in new file
        del g["tracks"]
        del g["instance_scores"]
        del g["point_scores"]
        del g["track_names"]
        del g["track_occupancy"]
        del g["tracking_scores"]

        # Create new datasets with modified data
        g.create_dataset("tracks", data=tracks)
        g.create_dataset("instance_scores", data=instance_scores)
        g.create_dataset("point_scores", data=point_scores)
        g.create_dataset("track_names", data=track_names, dtype=h5py.string_dtype(encoding='utf-8'))
        g.create_dataset("track_occupancy", data=track_occupancy)
        g.create_dataset("tracking_scores", data=tracking_scores)
        print('Saved a total of ' + str(len(track_names)) + ' tracks')
        print('File saved as ' + new_filename)
