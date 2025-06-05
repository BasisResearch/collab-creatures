import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pyro.infer import Predictive


def evaluate_performance(
    model, guide, predictors, outcome, num_samples=1000, plot=True
):

    predictive = Predictive(
        model=model, guide=guide, num_samples=num_samples, parallel=True
    )
    samples = predictive(predictors, outcome={key: None for key in outcome.keys()})

    predictions = samples[next(iter(outcome.keys()))]

    predictions_mean = predictions.squeeze().mean(dim=0)
    predictions_lower = predictions.squeeze().quantile(0.025, dim=0)
    predictions_upper = predictions.squeeze().quantile(0.975, dim=0)

    coverage = (
        (
            (predictions_lower <= outcome[next(iter(outcome.keys()))])
            & (outcome[next(iter(outcome.keys()))] <= predictions_upper)
        )
        .float()
        .mean()
    )

    observed_mean = outcome[next(iter(outcome.keys()))].mean()

    residuals = outcome[next(iter(outcome.keys()))] - predictions_mean

    mae = (torch.abs(residuals)).mean()

    rsquared = 1 - (
        torch.sum(residuals**2)
        / torch.sum((outcome[next(iter(outcome.keys()))] - observed_mean) ** 2)
    )

    if plot:

        fig, axs = plt.subplots(1, 2, figsize=(14, 10))

        axs[0].scatter(
            x=outcome[next(iter(outcome.keys()))],
            y=predictions_mean,
            s=6,
            alpha=0.5,
        )
        axs[0].set_title(
            "Ratio of outcomes within 95% CI: {:.2f}".format(coverage.item())
        )

        axs[0].set_xlabel("observed values")
        axs[0].set_ylabel("mean predicted values")

        axs[1].hist(residuals.detach().numpy(), bins=50)
        axs[1].set_title(f"Residuals, MAE: {mae.item():.2f}, R²: {rsquared.item():.2f}")

        axs[1].set_xlabel("residuals")
        axs[1].set_ylabel("count")

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        sns.despine()

        fig.suptitle("Model evaluation", fontsize=16)

        plt.show()
