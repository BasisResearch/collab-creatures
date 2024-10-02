from typing import Dict, Optional

import pyro
import pyro.distributions as dist
import torch


def continuous_contribution(
    continuous: Dict[str, torch.Tensor],
    child_name: str,
    leeway: float,
) -> torch.Tensor:

    contributions = torch.zeros(1)

    bias_continuous = pyro.sample(
            f"bias_continuous_{child_name}",
            dist.Normal(0.0, leeway),  # type: ignore
        )

    for key, value in continuous.items():
        

        weight_continuous = pyro.sample(
            f"weight_continuous_{key}_{child_name}",
            dist.Normal(0.0, leeway),  # type: ignore
        )

        contribution = bias_continuous + weight_continuous * value
        contributions = contribution + contributions

    return contributions


def add_linear_heteroskedastic_component(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    leeway: float,
    sigma_leeway: float,
    data_plate,
    observations: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    mean_prediction_child = continuous_contribution(
        child_continuous_parents, child_name, leeway
    )

    sigma_prediction_child = continuous_contribution(
        child_continuous_parents,
        f"{child_name}_sigma",
        sigma_leeway,
    )

    # sigma_prediction_child_transformed = (torch.abs(sigma_prediction_child - sigma_prediction_child.mean()) + 0.01)

    sigma_prediction_child_transformed = torch.nn.functional.softplus(
        sigma_prediction_child
    )

    # TODO think again about this if any related suspicion about performance
    # plt.hist(sigma_prediction_child.detach().numpy())
    # plt.title("Sigma")
    # plt.show()

    # plt.hist(sigma_prediction_child_transformed.detach().numpy())
    # plt.title("Transformed Sigma")
    # plt.show()

    assert sigma_prediction_child_transformed.min() > 0, "Sigma must be positive"

    with data_plate:

        child_observed = pyro.sample(  # type: ignore
            f"{child_name}",
            dist.Normal(mean_prediction_child, sigma_prediction_child_transformed),  # type: ignore
            obs=observations,
        )

    return child_observed


class HeteroskedasticLinear(pyro.nn.PyroModule):

    def __init__(
        self,
        predictors: Dict[str, torch.Tensor],
        outcome: Dict[str, torch.Tensor],
        leeway=0.6,
        sigma_leeway=0.2,
    ):
        super().__init__()
        self.predictors = predictors
        self.outcome = outcome
        self.leeway = leeway
        self.sigma_leeway = sigma_leeway
        self.outcome_name = next(iter(outcome.keys()))

        assert len(outcome) == 1, "Outcome must have a single key"

    def forward(self, predictors, outcome, n=None):

        if n is None:
            # take n from the first predictor
            n = next(iter(predictors.values())).shape[0]

        data_plate = pyro.plate("data", size=n, dim=-1)

        registered_predictors = {}

        with data_plate:

            for key in predictors.keys():
                registered_predictors[key] = pyro.sample(
                    key,
                    dist.Normal(0.0, 1),  # sd doesn't matter here,
                    # these should be either observed or potentially intervened
                    obs=predictors[key],
                )

        outcome_observed = add_linear_heteroskedastic_component(
            self.outcome_name,
            registered_predictors,
            self.leeway,
            self.sigma_leeway,
            data_plate,
            outcome[self.outcome_name],
        )

        return outcome_observed
