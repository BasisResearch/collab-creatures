from typing import Any, Callable, Dict, Optional

import pyro
import torch
from pyro.infer.enum import get_importance_trace


def compute_waic(
    model: Callable[..., Any],
    guide: Callable[..., Any],
    num_particles: int,
    max_plate_nesting: int,
    sites: Optional[list[str]] = None,
    *args: Any,
    **kwargs: Any
) -> Dict[str, Any]:

    def vectorize(fn: Callable[..., Any]) -> Callable[..., Any]:
        def _fn(*args: Any, **kwargs: Any) -> Any:
            with pyro.plate(
                "num_particles_vectorized", num_particles, dim=-max_plate_nesting
            ):
                return fn(*args, **kwargs)

        return _fn

    model_trace, _ = get_importance_trace(
        "flat", max_plate_nesting, vectorize(model), vectorize(guide), args, kwargs
    )

    def site_filter_is_observed(site_name: str) -> bool:
        return model_trace.nodes[site_name]["is_observed"]

    def site_filter_in_sites(site_name: str) -> bool:
        return sites is not None and site_name in sites

    if sites is None:
        site_filter = site_filter_is_observed
    else:
        site_filter = site_filter_in_sites

    observed_nodes = {
        name: node for name, node in model_trace.nodes.items() if site_filter(name)
    }

    log_p_post = {
        key: observed_nodes[key]["log_prob"].mean(dim=0)  # sum(axis = 0)/num_particles
        for key in observed_nodes.keys()
    }

    lppd = torch.stack([log_p_post[key] for key in log_p_post.keys()]).sum()

    var_log_p_post = {
        key: (observed_nodes[key]["log_prob"]).var(axis=0)
        for key in observed_nodes.keys()
    }

    p_waic = torch.stack([var_log_p_post[key] for key in var_log_p_post.keys()]).sum()

    waic = -2 * (lppd - p_waic)

    return {
        "waic": waic,
        "nodes": observed_nodes,
        "log_p_post": log_p_post,
        "var_log_p_post": var_log_p_post,
        "lppd": lppd,
        "p_waic": p_waic,
    }
