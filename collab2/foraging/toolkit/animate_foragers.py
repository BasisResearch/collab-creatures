import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

warnings.simplefilter(action="ignore", category=FutureWarning)


def plot_trajectories(df, title, ax=None, show_legend=True):
    unique_foragers = df["forager"].unique()
    
    if ax is None:
        _, ax = plt.subplots()



    for forager in unique_foragers:
        df_forager = df[df["forager"] == forager]
        (line,) = ax.plot(df_forager["x"], df_forager["y"])
        init_loc = df_forager[df_forager.time == 0]
        # use same color as the trajectory
        ax.scatter(
            init_loc["x"],
            init_loc["y"],
            color=line.get_color(),
            s=50,
            marker="o",
            label=f"Forager {forager}: initial",
        )
        final_loc = df_forager[df_forager.time == df_forager.time.max()]
        ax.scatter(
            final_loc["x"],
            final_loc["y"],
            color=line.get_color(),
            s=50,
            marker="x",
            label=f"Forager {forager}: final",
        )

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_axis_off()
    
    if show_legend:
        ax.legend()
    
    ax.set_title(f"Trajectories: {title}", fontsize=16)
    return ax


def plot_distances(distances, title=""):
    distances_list = [
        distance
        for sublist in distances
        for df in sublist
        for distance in df["distance"].tolist()
    ]

    distances_list = list(filter(lambda x: x != 0, distances_list))
    fig = px.histogram(
        distances_list,
        template="presentation",
        width=700,
        title=f"Distances: {title}",
        labels={"value": "inter-bird distance (grid units)"},
        opacity=0.4,
        nbins=60,
    )

    fig.update_layout(showlegend=False)
    return fig


def animate_foragers(
    sim,
    width=800,
    height=800,
    point_size=15,
    plot_rewards=True,
    plot_traces=False,
    plot_visibility=0,
    plot_proximity=0,
    plot_communicate=0,
    plot_velocity=0,
    trace_multiplier=10,
    visibility_multiplier=10,
    proximity_multiplier=10,
    communicate_multiplier=10,
    velocity_multiplier=10,
    color_by_state=False,
    produce_object=False,
):
    if plot_rewards:
        rew = sim.rewardsDF.copy()
        if color_by_state:
            rew["state"] = "reward"
        else:
            rew["forager"] = "reward"
        df = pd.concat([sim.foragersDF, rew])

    else:
        df = sim.foragersDF.copy()

    if plot_traces:
        tr = sim.tracesDF.copy()
        tr["forager"] = "trace"
        df = pd.concat([df, tr])

    if plot_visibility > 0:
        vis = sim.visibilityDF.copy()
        vis = vis[vis["forager"] == plot_visibility]
        vis["who"] = vis["forager"]
        vis["forager"] = "visibility"
        df = pd.concat([df, vis])

    if plot_proximity > 0:
        prox = sim.proximityDF.copy()
        prox = prox[prox["forager"] == plot_proximity]
        prox["who"] = prox["forager"]
        prox["forager"] = "proximity"
        df = pd.concat([df, prox])

    if plot_communicate > 0:
        com = sim.communicatesDF.copy()
        com = com[com["forager"] == plot_communicate]
        com["who"] = com["forager"]
        com["forager"] = "communicate"
        com = com.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df = pd.concat([com, df], axis=0, ignore_index=True, verify_integrity=True)

    if plot_velocity > 0:
        vel = sim.velocity_scoresDF.copy()
        vel = vel[vel["forager"] == plot_velocity]
        vel["who"] = vel["forager"]
        vel["forager"] = "velocity"
        vel = vel.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df = pd.concat([vel, df], axis=0, ignore_index=True, verify_integrity=True)

    if not color_by_state:
        fig = px.scatter(df, x="x", y="y", animation_frame="time", color="forager")
    else:
        fig = px.scatter(df, x="x", y="y", animation_frame="time", color="state")

    fig.update_layout(
        template="presentation",
        xaxis=dict(
            range=[-1, sim.grid_size + 1],
            showgrid=False,
            zeroline=False,
            ticks="",
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            range=[-1, sim.grid_size + 1],
            showgrid=False,
            zeroline=False,
            ticks="",
            showticklabels=False,
            title="",
            scaleanchor="x",  # This makes the y-axis scale to match the x-axis
        ),
        autosize=False,
        width=width,
        height=height,
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0

    fig.update_traces(marker=dict(size=point_size))

    for t in range(0, len(fig.frames)):
        for trace in fig.frames[t].data:
            if trace.name.isdigit():
                trace.marker.symbol = "square"
                trace.marker.size = 14
                trace.marker.line = dict(width=3)
                trace.marker.opacity = 0.8

    if plot_rewards:
        fig.update_traces(
            showlegend=False,
            marker=dict(symbol="square", color="yellow"),
            selector=dict(name="reward"),
        )

        for frame in fig.frames:
            for trace in frame.data:
                if trace.name == "reward":
                    trace.marker.symbol = "square"
                    trace.marker.color = "yellow"
                    trace.showlegend = False

    if plot_velocity > 0:
        fig.update_traces(showlegend=False, selector=dict(name="velocity"))

        for t in range(0, len(fig.frames)):
            selected_rows = vel[(vel["time"] == t + 1)]
            for trace in fig.frames[t].data:
                if trace.name == "velocity":
                    trace.marker.symbol = "circle"
                    trace.marker.color = "red"
                    trace.showlegend = False
                    trace.marker.size = (
                        selected_rows["velocity_score"] * velocity_multiplier
                    )
                    trace.marker.opacity = 0.3

    if plot_communicate > 0:
        fig.update_traces(showlegend=False, selector=dict(name="communicate"))

        for t in range(0, len(fig.frames)):
            selected_rows = com[(com["time"] == t + 1)]
            for trace in fig.frames[t].data:
                if trace.name == "communicate":
                    trace.marker.symbol = "circle"
                    trace.marker.color = "red"
                    trace.showlegend = False
                    trace.marker.size = (
                        selected_rows["communicate"] * communicate_multiplier
                    )
                    trace.marker.opacity = 0.3

    if plot_traces:
        fig.update_traces(showlegend=False, selector=dict(name="trace"))

        for t in range(0, len(fig.frames)):
            selected_rows = sim.tracesDF[sim.tracesDF["time"] == t + 1]
            for trace in fig.frames[t].data:
                if trace.name == "trace":
                    trace.marker.symbol = "circle"
                    trace.marker.color = "orange"
                    trace.showlegend = False
                    trace.marker.size = selected_rows["trace"] * trace_multiplier
                    trace.marker.opacity = 0.3

    if plot_visibility > 0:
        fig.update_traces(showlegend=False, selector=dict(name="visibility"))

        for t in range(0, len(fig.frames)):
            selected_rows = vis[(vis["time"] == t + 1)]
            for trace in fig.frames[t].data:
                if trace.name == "visibility":
                    trace.marker.symbol = "circle"
                    trace.marker.color = "gray"
                    trace.showlegend = False
                    trace.marker.size = (
                        selected_rows["visibility"] * visibility_multiplier
                    )
                    trace.marker.opacity = 0.3

    if plot_proximity > 0:
        color_scale = "Greys"

        fig.update_traces(showlegend=False, selector=dict(name="proximity"))

        for t in range(0, len(fig.frames)):
            selected_rows = prox[(prox["time"] == t + 1)]
            for trace in fig.frames[t].data:
                if trace.name == "proximity":
                    trace.marker.symbol = "circle"
                    # trace.marker.color = "red"
                    trace.showlegend = False
                    trace.marker.color = (
                        selected_rows["proximity"] * proximity_multiplier
                    )
                    trace.marker.colorscale = color_scale
                    trace.marker.size = 5
                    trace.marker.opacity = 0.6

    fig = go.Figure(
        data=fig["frames"][0]["data"],
        frames=fig["frames"],
        layout=fig.layout,
    )

    if produce_object:
        return fig
    else:
        fig.show()


def visualise_forager_predictors(
    outcome: torch.Tensor,
    predictors: List[torch.Tensor],
    predictor_names: List[str],
    outcome_name: str,
    sampling_rate: float = 1.0,
    titles=None,
):
    def sample_tensor(tensor, sampling_rate):
        sample_size = int(sampling_rate * len(tensor))
        return np.random.choice(tensor, size=sample_size, replace=False)

    def custom_copy(tr):
        if isinstance(tr, torch.Tensor):
            return tr.clone()
        else:
            return tr.copy()

    if sampling_rate != 1:
        outcome_sub = sample_tensor(outcome, sampling_rate)
        predictors_sub = [
            sample_tensor(predictor, sampling_rate) for predictor in predictors
        ]
    else:
        outcome_sub = custom_copy(outcome)
        predictors_sub = [custom_copy(predictor) for predictor in predictors]

    df = pd.DataFrame({"outcome": outcome_sub})
    for name, predictor_sub in zip(predictor_names, predictors_sub):
        df[name] = predictor_sub

    for idx, name in enumerate(predictor_names):
        fig = px.scatter(
            df,
            x=name,
            y="outcome",
            opacity=0.3,
            template="presentation",
            width=700,
        )

        title = titles[idx] if titles else name

        fig.update_layout(
            title=title.capitalize(),
            xaxis_title=name,
            yaxis_title=outcome_name,
        )

        fig.update_traces(marker={"size": 4})
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show()


def plot_coefs(
    selected_samples: Dict[str, torch.Tensor],
    title: str,
    nbins=20,
    ann_start_y=100,
    ann_break_y=50,
    generate_object=False,
):

    for key in selected_samples.keys():
        selected_samples[key] = selected_samples[key].flatten()

    samplesDF = pd.DataFrame(selected_samples)
    samplesDF_medians = samplesDF.median(axis=0)

    fig_coefs = px.histogram(
        samplesDF,
        template="presentation",
        opacity=0.4,
        labels={"variable": "coefficient"},
        width=700,
        title=title,
        nbins=nbins,
        marginal="rug",
        barmode="overlay",
    )

    color_scale = px.colors.qualitative.Alphabet

    for i, median_value in enumerate(samplesDF_medians):
        color = color_scale[i % len(color_scale)]
        fig_coefs.add_vline(
            x=median_value,
            line_dash="dash",
            line_color=color,
            name=f"Median ({samplesDF_medians.iloc[i]})",
        )

        fig_coefs.add_annotation(
            x=samplesDF_medians.iloc[i],
            y=ann_start_y
            + ann_break_y * i,  # Adjust the vertical position of the label
            text=f"{samplesDF_medians.iloc[i]:.2f}",
            showarrow=False,
            bordercolor="black",
            borderwidth=0.5,
            bgcolor="white",
            opacity=0.8,
        )

    fig_coefs.update_layout(
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",  # Anchor the legend to the top of the container
            y=-0.25,  # Position it below the plot
            xanchor="center",  # Center it horizontally
            x=0.5,  # Center it horizontally in the plot
            title_text="Legend",  # Optional: Title for the legend
        )
    )

    fig_coefs.update_traces(marker=dict(line=dict(width=2, color="Black")))

    fig_coefs.show()

    if generate_object:
        return fig_coefs
