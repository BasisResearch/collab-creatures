from .access import generate_access_predictor  # noqa: F401
from .animate_foragers import (  # noqa: F401
    animate_foragers,
    plot_coefs,
    plot_distances,
    plot_trajectories,
    visualise_forager_predictors,
)
from .communicates import generate_communicates  # noqa: F401
from .communication import generate_communication_predictor  # noqa: F401
from .derive import (  # noqa: F401
    derive_predictors_and_scores,
    get_list_of_predictors,
    get_list_of_scores,
)

#  TODO resolve if any major differences between them and potentially unify
from .dynamical_utils import tensorize_and_dump_count_data  # noqa: F401
from .dynamical_utils import (  # run_svi_inference # noqa: F401;
    add_ring,
    ds_uncertainty_plot,
    plot_ds_trajectories,
)
from .evaluate import evaluate_performance  # noqa: F401
from .filtering import constraint_filter_nearest, filter_by_distance  # noqa: F401
from .food import generate_food_predictor  # noqa: F401
from .inference import run_svi_inference  # type: ignore # noqa: F401
from .inference import (  # noqa: F401
    get_samples,
    prep_data_for_inference,
    prep_DF_data_for_inference,
    summary,
)
from .local_windows import (  # noqa: F401
    _generate_local_windows,
    _get_grid,
    generate_local_windows,
)
from .models import (  # noqa: F401
    HeteroskedasticLinear,
    add_linear_heteroskedastic_component,
    continuous_contribution,
)
from .next_step_score import (  # noqa: F401
    _generate_nextStep_score,
    generate_nextStep_score,
)
from .proximity import (  # noqa: F401; foragers_to_forager_distances,
    generate_proximity_predictor,
)
from .subsampling import (  # noqa: F401
    rescale_to_grid,
    sample_time_slices,
    subsample_frames_constant_frame_rate,
    subset_frames_evenly_spaced,
)
from .utils import (  # noqa: F401
    dataObject,
    distances_and_peaks,
    foragers_to_forager_distances,
    update_rewards,
)
from .velocity import (  # noqa: F401
    _add_velocity,
    _generic_velocity_predictor,
    _velocity_predictor_contribution,
    generate_pairwiseCopying_predictor,
    generate_vicsek_predictor,
)
from .visibility import (  # noqa: F401
    construct_visibility,
    filter_by_visibility,
    visibility_vs_distance,
)
from .visualization import animate_predictors, plot_predictor  # noqa: F401

# from .trace import rewards_to_trace, rewards_trace  # noqa: F401
