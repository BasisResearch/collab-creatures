import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def animate_birds(
    sim, width=800, height=800, point_size=15, plot_rewards=True
):
    if plot_rewards:
        rew = sim.rewardsDF
        rew["bird"] = "reward"
        df = pd.concat([sim.birdsDF, sim.rewardsDF])

    else:
        df = sim.birdsDF

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
    fig.update_traces(showlegend=False, selector=dict(name="reward"))
    # fig.update_traces(transition=False, selector=dict(name='reward'))
    fig.update_traces(
        marker=dict(symbol="square", color="yellow"),
        selector=dict(name="reward"),
    )

    for frame in fig.frames:
        for trace in frame.data:
            if trace.name == "reward":
                trace.marker.symbol = "square"
                trace.marker.color = "yellow"
                trace.showlegend = False

    fig.show()
