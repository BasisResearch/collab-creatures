import os
import random

import numpy as np
import pandas as pd
import pytest

from collab.foraging import random_hungry_followers as rhf
from collab.foraging import toolkit as ft
from collab.utils import find_repo_root

root = find_repo_root()


def test_random_derivation():
    random.seed(22)
    np.random.seed(22)

    random_foragers_sim = rhf.RandomForagers(
        grid_size=40,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_foragers=3,
        num_frames=5,
        num_rewards=15,
        grab_range=3,
    )

    random_foragers_sim()
    random_foragers_derived = ft.derive_predictors(random_foragers_sim, dropna=False)
    module_dir = os.path.dirname(__file__)
    path = os.path.join(module_dir, "random_test_data.csv")
    rhf_test_data = pd.read_csv(path)

    assert random_foragers_derived.derivedDF.shape[0] == rhf_test_data.shape[0]
    assert random_foragers_derived.derivedDF.shape[1] == 15


@pytest.mark.filterwarnings(
    "ignore:Behavior when concatenating bool-dtype and numeric-dtype arrays is deprecated"
)
def test_hungry_derivation():
    random.seed(22)
    np.random.seed(22)

    hungry_sim = rhf.Foragers(
        grid_size=40, num_foragers=3, num_frames=10, num_rewards=10, grab_range=3
    )

    hungry_sim()

    hungry_sim = rhf.add_hungry_foragers(
        hungry_sim, num_hungry_foragers=3, rewards_decay=0.3, visibility_range=6
    )

    hungry_sim_derived = ft.derive_predictors(hungry_sim, dropna=False)

    module_dir = os.path.dirname(__file__)
    path = os.path.join(module_dir, "hungry_test_data.csv")
    hungry_test_data = pd.read_csv(path)

    assert hungry_sim_derived.derivedDF.shape[0] == hungry_test_data.shape[0]
    assert hungry_sim_derived.derivedDF.shape[1] == 15 

