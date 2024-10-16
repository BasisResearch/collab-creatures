import logging

import pandas as pd

from collab.foraging.toolkit.communicates import generate_communicates
from collab.foraging.toolkit.how_far import add_how_far_squared_scaled
from collab.foraging.toolkit.proximity import generate_proximity_score
from collab.foraging.toolkit.trace import rewards_to_trace
from collab.foraging.toolkit.utils import generate_grid
from collab.foraging.toolkit.velocity import (
    add_velocities_to_data_object,
    generate_velocity_scores,
)
from collab.foraging.toolkit.visibility import construct_visibility

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
derivation_logger = logging.getLogger(__name__)


# import foraging_toolkit as ft


def derive_predictors(
    sim,
    rewards_decay=0.5,
    visibility_range=10,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    dropna=True,
    time_shift=0,
    sampling_rate=1,
    random_seed=42,
    # communicates
    generate_communicates_indicator=False,
    communicate_visibility_restriction="invisible",
    communicate_filter_by_on_reward=False,
    communicate_info_time_decay=3,
    communicate_info_spatial_decay=0.15,
    communicate_finders_tolerance=2,
    # velocity
    generate_velocity_indicator=False,
    velocity_time_decay=3,
    velocity_spatial_decay=0.15,
    velocity_visibility_restriction="visible",
):
    sim.communicate_visibility_restriction = communicate_visibility_restriction
    sim.communicate_filter_by_on_reward = communicate_filter_by_on_reward

    sim.velocity_time_decay = velocity_time_decay
    sim.velocity_spatial_decay = velocity_spatial_decay
    sim.velocity_visibility_restriction = velocity_visibility_restriction

    grid = generate_grid(sim.grid_size)

    grid = grid.sample(frac=sampling_rate, random_state=random_seed)

    sim.grid = grid

    tr = rewards_to_trace(
        sim.rewards,
        sim.grid_size,
        sim.num_frames,
        rewards_decay,
        time_shift=time_shift,
        grid=grid,
    )

    sim.traces = tr["traces"]
    sim.tracesDF = tr["tracesDF"]
    derivation_logger.info("traces done")

    vis = construct_visibility(
        sim.foragers,
        sim.grid_size,
        visibility_range=visibility_range,
        time_shift=time_shift,
        grid=grid,
    )
    sim.visibility_range = visibility_range
    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]
    derivation_logger.info("visibility done")

    prox = generate_proximity_score(
        sim.foragers,
        sim.visibility,
        visibility_range=visibility_range,
        getting_worse=getting_worse,
        optimal=optimal,
        proximity_decay=proximity_decay,
        time_shift=time_shift,
    )
    sim.getting_worse = getting_worse
    sim.optimal = optimal
    sim.proximity_decay = proximity_decay

    sim.proximity = prox["proximity"]
    sim.proximityDF = prox["proximityDF"]
    derivation_logger.info("proximity done")

    add_how_far_squared_scaled(sim)
    derivation_logger.info("how_far done")

    sim.derivedDF = (
        sim.tracesDF.merge(sim.visibilityDF, how="inner")
        .merge(sim.proximityDF, how="inner")
        .merge(sim.how_farDF, how="inner")
    )
    derivation_logger.info("derivedDF done")

    if generate_communicates_indicator:
        derivation_logger.info("starting to generate communicates")
        com = generate_communicates(
            sim,
            communicate_info_time_decay,
            communicate_info_spatial_decay,
            finders_tolerance=communicate_finders_tolerance,
            time_shift=time_shift,
            grid=grid,
            visibility_restriction=sim.communicate_visibility_restriction,
            filter_by_on_reward=sim.communicate_filter_by_on_reward,
        )
        sim.communicates = com["communicates"]
        sim.communicatesDF = com["communicatesDF"]

        sim.derivedDF = sim.derivedDF.merge(sim.communicatesDF, how="left")

        sim.derivedDF["communicate"].fillna(0, inplace=True)
        sim.communicatesDF.loc[:, "time"] = sim.communicatesDF["time"] - time_shift

        derivation_logger.info("communicates done")

    if generate_velocity_indicator:
        derivation_logger.info("starting to generate velocity")

        add_velocities_to_data_object(sim)

        vs = generate_velocity_scores(
            sim,
            velocity_time_decay=sim.velocity_time_decay,
            velocity_spatial_decay=sim.velocity_spatial_decay,
            time_shift=time_shift,
            grid=grid,
            visibility_restriction=sim.velocity_visibility_restriction,
        )

        sim.velocity_scores = vs["velocity_scores"]
        sim.velocity_scoresDF = vs["velocity_scoresDF"]

        sim.derivedDF = sim.derivedDF.merge(sim.velocity_scoresDF, how="left")

        sim.derivedDF["velocity_score"].fillna(0, inplace=True)
        sim.velocity_scoresDF.loc[:, "time"] = (
            sim.velocity_scoresDF["time"] - time_shift
        )

        derivation_logger.info("velocity done")

    pd.set_option("mode.chained_assignment", None)
    sim.rewardsDF.loc[:, "time"] = sim.rewardsDF["time"] - time_shift
    sim.foragersDF.loc[:, "time"] = sim.foragersDF["time"] - time_shift
    sim.tracesDF.loc[:, "time"] = sim.tracesDF["time"] - time_shift
    sim.visibilityDF.loc[:, "time"] = sim.visibilityDF["time"] - time_shift
    sim.proximityDF.loc[:, "time"] = sim.proximityDF["time"] - time_shift
    sim.how_farDF.loc[:, "time"] = sim.how_farDF["time"] - time_shift
    sim.derivedDF.loc[:, "time"] = sim.derivedDF["time"] - time_shift

    if dropna:
        sim.derivedDF.dropna(inplace=True)

    return sim
