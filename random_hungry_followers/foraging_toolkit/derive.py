import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
derivation_logger = logging.getLogger(__name__)


# Create a logger for your module (use a unique name for each module)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)
import foraging_toolkit as ft


def derive_predictors(
    sim,
    rewards_decay=0.5,
    visibility_range=10,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    generate_communicates=False,
    info_time_decay=3,
    info_spatial_decay=0.15,
    finders_tolerance=2,
    time_shift=0,
):
    tr = ft.rewards_to_trace(
        sim.rewards,
        sim.grid_size,
        sim.num_frames,
        rewards_decay,
        time_shift=time_shift,
    )
    sim.traces = tr["traces"]
    sim.tracesDF = tr["tracesDF"]
    derivation_logger.info("traces done")

    vis = ft.construct_visibility(
        sim.birds,
        sim.grid_size,
        visibility_range=visibility_range,
        time_shift=time_shift,
    )
    sim.visibility_range = visibility_range
    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]
    derivation_logger.info("visibility done")

    prox = ft.generate_proximity_score(
        sim.birds,
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

    ft.add_how_far_squared_scaled(sim)
    derivation_logger.info("how_far done")

    sim.derivedDF = (
        sim.tracesDF.merge(sim.visibilityDF, how="inner")
        .merge(sim.proximityDF, how="inner")
        .merge(sim.how_farDF, how="inner")
    )
    derivation_logger.info("derivedDF done")

    if generate_communicates:
        derivation_logger.info("starting to generate communicates")
        com = ft.generate_communicates(
            sim,
            info_time_decay,
            info_spatial_decay,
            finders_tolerance=finders_tolerance,
            time_shift=time_shift,
        )
        sim.communicates = com["communicates"]
        sim.communicatesDF = com["communicatesDF"]

        sim.derivedDF = sim.derivedDF.merge(sim.communicatesDF, how="left")

        sim.derivedDF["communicate"].fillna(0, inplace=True)

        # mean_communicate = sim.communicatesDF["communicate"].mean()
        # std_communicate = sim.communicatesDF["communicate"].std()

        # sim.derivedDF["communicate_standardized"] = (
        #     sim.derivedDF["communicate"] - sim.derivedDF["communicate"].mean()
        # ) / sim.derivedDF["communicate"].std()

        derivation_logger.info("communicates done")

    pd.set_option("mode.chained_assignment", None)
    sim.rewardsDF.loc[:, "time"] = sim.rewardsDF["time"] - time_shift
    sim.birdsDF.loc[:, "time"] = sim.birdsDF["time"] - time_shift
    sim.tracesDF.loc[:, "time"] = sim.tracesDF["time"] - time_shift
    sim.visibilityDF.loc[:, "time"] = sim.visibilityDF["time"] - time_shift
    sim.proximityDF.loc[:, "time"] = sim.proximityDF["time"] - time_shift
    sim.how_farDF.loc[:, "time"] = sim.how_farDF["time"] - time_shift
    sim.communicatesDF.loc[:, "time"] = sim.communicatesDF["time"] - time_shift
    sim.derivedDF.loc[:, "time"] = sim.derivedDF["time"] - time_shift

    # TODO: maybe ignore outliers resulting from bird crowding?
    # sim.derivedDF = sim.derivedDF[sim.derivedDF["proximity"] > 0]

    return sim
