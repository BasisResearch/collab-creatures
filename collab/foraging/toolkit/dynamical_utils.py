import numpy as np


def add_ring(
    df,
    name,
    x0,
    y0,
    outside_radius=10,
    inside_radius=0,
    keep_distance=False,
    divide_by_side=False,
):
    df_c = df.copy()
    if "state" not in df.columns:
        df_c["state"] = "unclassified"

    df_c["_distance"] = np.sqrt((df_c["x"] - x0) ** 2 + (df_c["y"] - y0) ** 2)

    _condition = (
        (df_c["state"] == "unclassified")
        & (df_c["_distance"] < outside_radius)
        & (df_c["_distance"] > inside_radius)
    )
    if not divide_by_side:
        df_c.loc[_condition, "state"] = name
    else:
        df_c.loc[_condition & (df_c["x"] >= x0), "state"] = f"{name}_r"
        df_c.loc[_condition & (df_c["x"] < x0), "state"] = f"{name}_l"

    if not keep_distance:
        df_c.drop("_distance", axis=1, inplace=True)

    return df_c
