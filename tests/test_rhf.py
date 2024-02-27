import os
import random

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from collab.foraging import random_hungry_followers as rhf
from collab.foraging import toolkit as ft


def test_rhf_derivation():
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
    path = os.path.join(module_dir, "rhf_test_data.csv")
    rhf_test_data = pd.read_csv(path)

    assert_frame_equal(random_foragers_derived.derivedDF, rhf_test_data)
