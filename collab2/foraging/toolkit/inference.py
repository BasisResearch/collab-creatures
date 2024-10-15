import logging
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean


def prep_data_for_inference(
    sim_derived, predictors: List[str], outcome_vars: str, subsample_rate: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    if isinstance(outcome_vars, str):
        outcome_list = [outcome_vars]
    else:
        outcome_list = outcome_vars

    df = sim_derived.derivedDF[predictors + outcome_list].copy()

    # assert no NaNs in df
    assert df.notna().all().all(), "Dataframe contains NaN values"

    # Apply subsampling
    if subsample_rate < 1.0:
        df = df.sample(frac=subsample_rate).reset_index(drop=True)

    predictor_tensors = {
        key: torch.tensor(df[key].values, dtype=torch.float32) for key in predictors
    }
    outcome_tensors = {
        key: torch.tensor(df[key].values, dtype=torch.float32) for key in outcome_list
    }

    # print size
    print("Sample size:", len(df))

    return predictor_tensors, outcome_tensors


def prep_DF_data_for_inference(
    DF, predictors: List[str], outcome_vars: str, subsample_rate: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    if isinstance(outcome_vars, str):
        outcome_list = [outcome_vars]
    else:
        outcome_list = outcome_vars

    df = DF[predictors + outcome_list].copy()

    # assert no NaNs in df
    assert df.notna().all().all(), "Dataframe contains NaN values"

    # Apply subsampling
    if subsample_rate < 1.0:
        df = df.sample(frac=subsample_rate).reset_index(drop=True)

    predictor_tensors = {
        key: torch.tensor(df[key].values, dtype=torch.float32) for key in predictors
    }
    outcome_tensors = {
        key: torch.tensor(df[key].values, dtype=torch.float32) for key in outcome_list
    }

    # print size
    print("Sample size:", len(df))

    return predictor_tensors, outcome_tensors












def summary(samples, sites):
    site_stats = {}
    for site_name, values in samples.items():
        if site_name in sites:
            marginal_site = pd.DataFrame(values)
            describe = marginal_site.describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            ).transpose()
            site_stats[site_name] = describe[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
    return site_stats


def run_svi_inference(
    model,
    verbose=True,
    lr=0.03,
    vi_family=AutoMultivariateNormal,
    guide=None,
    n_steps=100,
    ylim=None,
    plot=True,
    **model_kwargs,
):
    losses = []
    if guide is None:
        guide = vi_family(model, init_loc_fn=init_to_mean)
    elbo = pyro.infer.Trace_ELBO()(model, guide)

    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)

    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        losses.append(loss.item())
        adam.step()
        if (step % 200 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))

    if plot:
        plt.plot(losses)
        if ylim:
            plt.ylim(ylim)
        plt.show()

    return guide


def get_samples(
    model,
    predictors,
    outcome,
    num_svi_iters,
    num_samples,
    plot = True,
    verbose = True,
):

    logging.info(f"Starting SVI inference with {num_svi_iters} iterations.")
    start_time = time.time()
    pyro.clear_param_store()
    guide = run_svi_inference(
        model, n_steps=num_svi_iters, predictors=predictors, outcome=outcome, 
        plot = plot, verbose = verbose
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("SVI inference completed in %.2f seconds.", elapsed_time)

    predictive = pyro.infer.Predictive(
        model=model, guide=guide, num_samples=num_samples, parallel=True
    )

    samples = {
        k: v.flatten().reshape(num_samples, -1).detach().cpu().numpy()
        for k, v in predictive(predictors, outcome).items()
        if k != "obs"
    }

    sites = [
        key
        for key in samples.keys()
        if (key.startswith("weight") and not key.endswith("sigma"))
    ]
    print(sites)

    print("Coefficient marginals:")
    for site, values in summary(samples, sites).items():
        print("Site: {}".format(site))
        print(values, "\n")

    return {"samples": samples, "guide": guide, "predictive": predictive, 
            "summaries":
            summary(samples, sites)}
