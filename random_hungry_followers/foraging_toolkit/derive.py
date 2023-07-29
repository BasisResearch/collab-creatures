import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd


import sys
import os

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
):
    tr = ft.rewards_to_trace(
        sim.rewards,
        sim.grid_size,
        sim.num_frames,
        rewards_decay,
    )
    sim.traces = tr["traces"]
    sim.tracesDF = tr["tracesDF"]

    vis = ft.construct_visibility(
        sim.birds, sim.grid_size, visibility_range=visibility_range
    )
    sim.visibility_range = visibility_range
    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    prox = ft.generate_proximity_score(
        sim.birds,
        sim.visibility,
        visibility_range=visibility_range,
        getting_worse=getting_worse,
        optimal=optimal,
        proximity_decay=proximity_decay,
    )
    sim.getting_worse = getting_worse
    sim.optimal = optimal
    sim.proximity_decay = proximity_decay

    sim.proximity = prox["proximity"]
    sim.proximityDF = prox["proximityDF"]

    ft.add_how_far_squared_scaled(sim)

    sim.derivedDF = (
        sim.tracesDF.merge(sim.visibilityDF, how="inner")
        .merge(sim.proximityDF, how="inner")
        .merge(sim.how_farDF, how="inner")
    )

    # TODO: ignore outliers resulting from bird crowding?
    sim.derivedDF = sim.derivedDF[sim.derivedDF["proximity"] > 0]

    return sim
