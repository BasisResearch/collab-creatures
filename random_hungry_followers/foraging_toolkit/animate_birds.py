import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt


def animate_birds(
    sim,
    width=800,
    height=800,
    point_size=15,
    plot_rewards=True,
    plot_traces=False,
    plot_visibility=0,
    plot_proximity=0,
    plot_communicate=0,
    trace_multiplier=10,
    visibility_multiplier=10,
    proximity_multiplier=10,
    communicate_multiplier=10,
):
    if plot_rewards:
        rew = sim.rewardsDF.copy()
        rew["bird"] = "reward"
        df = pd.concat([sim.birdsDF, rew])

    else:
        df = sim.birdsDF.copy()

    if plot_traces:
        tr = sim.tracesDF.copy()
        tr["bird"] = "trace"
        df = pd.concat([df, tr])

    if plot_visibility > 0:
        vis = sim.visibilityDF.copy()
        vis = vis[vis["bird"] == plot_visibility]
        vis["who"] = vis["bird"]
        vis["bird"] = "visibility"
        df = pd.concat([df, vis])

    if plot_proximity > 0:
        prox = sim.proximityDF.copy()
        prox = prox[prox["bird"] == plot_proximity]
        prox["who"] = prox["bird"]
        prox["bird"] = "proximity"
        df = pd.concat([df, prox])

    if plot_communicate > 0:
        com = sim.communicatesDF.copy()
        com = com[com["bird"] == plot_communicate]
        com["who"] = com["bird"]
        com["bird"] = "communicate"
        com = com.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df = pd.concat(
            [com, df], axis=0, ignore_index=True, verify_integrity=True
        )

    fig = px.scatter(df, x="x", y="y", animation_frame="time", color="bird")

    fig.update_layout(
        template="plotly_dark",
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
                    trace.marker.size = (
                        selected_rows["trace"] * trace_multiplier
                    )
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
        color_scale = "Purples"

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

    fig.show()


def visualise_bird_predictors(tr, prox, hf, com=None):
    if com is not None:
        df = pd.DataFrame(
            {
                "trace": tr,
                "proximity": prox,
                "communicate": com,
                "how_far_score": hf,
            }
        )
    else:
        df = pd.DataFrame(
            {"trace": tr, "proximity": prox, "how_far_score": hf}
        )

    fig = px.scatter(
        df,
        x="trace",
        y="how_far_score",
        opacity=0.3,
        template="plotly_dark",
    )
    fig.update_layout(
        title="Trace",
        xaxis_title="trace",
        yaxis_title="how far score",
    )

    fig2 = px.scatter(
        df,
        x="proximity",
        y="how_far_score",
        opacity=0.3,
        template="plotly_dark",
    )
    fig2.update_layout(
        title="Proximity",
        xaxis_title="proximity",
        yaxis_title="how far score",
    )

    fig.update_traces(marker={"size": 4})
    fig2.update_traces(marker={"size": 4})

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)

    fig.show()
    fig2.show()

    if com is not None:
        fig3 = px.scatter(
            df,
            title = "Communication",
            x="communicate",
            y="how_far_score",
            opacity=0.3,
            template="plotly_dark",
        )
        fig3.update_traces(marker={"size": 4})
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=False)

        fig3.show()
