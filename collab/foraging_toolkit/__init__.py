from .utils import generate_grid, update_rewards, object_from_data

from .trace import (
    rewards_trace,
    rewards_to_trace,
)


from .random_birds import Birds, RandomBirds

from .hungry_birds import add_hungry_birds
from .follower_birds import add_follower_birds

from .visibility import visibility_vs_distance, construct_visibility

from .proximity import proximity_score, birds_to_bird_distances, generate_proximity_score

# distances_and_peaks

from .communicates import generate_communicates

from .how_far import add_how_far_squared_scaled


from .derive import derive_predictors

from .inference import (
    prep_data_for_communicators_inference,
    prep_data_for_robust_inference,
    get_tensorized_data,
    get_svi_results,
    sample_and_plot_coef,
    model_sigmavar_com,
    svi_training,
    svi_prediction,
    summary,
    mcmc_training,
    normalize,
)

from .animate_birds import animate_birds, visualise_bird_predictors

from .locust import (
    load_and_clean_locust,
    locust_object_from_data,
)

from .subsampling import subsample_frames_evenly_spaced, rescale_to_grid, sample_time_slices


from .central_park import (
    cp_birds_to_bird_distances,
    cp_distances_and_peaks,
    cp_generate_visibility,
)
