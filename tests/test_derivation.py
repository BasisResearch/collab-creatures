import os
import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

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

    assert_frame_equal(random_foragers_derived.derivedDF, rhf_test_data)


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

    assert_frame_equal(hungry_sim_derived.derivedDF, hungry_test_data)


def test_followers_derivation():
    random.seed(22)
    np.random.seed(22)

    follower_sim = rhf.Foragers(
        grid_size=60, num_foragers=3, num_frames=10, num_rewards=10, grab_range=3
    )
    follower_sim()

    follower_sim = rhf.add_follower_foragers(
        follower_sim, num_follower_foragers=3, proximity_decay=0.3, visibility_range=6
    )

    follower_sim_derived = ft.derive_predictors(
        follower_sim, getting_worse=0.5, optimal=3, visibility_range=6, dropna=False
    )

    module_dir = os.path.dirname(__file__)
    path = os.path.join(module_dir, "followers_test_data.csv")
    followers_test_data = pd.read_csv(path)

    assert_frame_equal(follower_sim_derived.derivedDF, followers_test_data)
