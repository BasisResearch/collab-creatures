import pandas as pd


def summary(samples, sites):
    if sites is None:
        sites = [site_name for site_name in samples.keys()]

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


def mcmc_training(model, num_samples, sites=None, *args):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_samples // 4)
    mcmc.run(*args)

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    if sites is None:
        sites = [site_name for site_name in hmc_samples.keys()]

    for site, values in summary(hmc_samples).items():
        if site in ["t", "p"]:
            print("Site: {}".format(site))
            print(values, "\n")

    return hmc_samples
