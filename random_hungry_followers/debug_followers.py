import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Go down one level from the current directory (replace 'folder_name' with the actual name of the folder)
folder_path = os.path.join(current_dir, "random_hungry_followers")

# Add the folder path to sys.path
sys.path.insert(0, folder_path)
# print(sys.path)

import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import pyro
import foraging_toolkit as ft
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.optim as optim
from pyro.nn import PyroModule
from pyro.infer.autoguide import (
    AutoNormal,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    init_to_mean,
    init_to_value,
)
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive
from pyro.infer import MCMC, NUTS

import os
import logging
import time

logging.basicConfig(format="%(message)s", level=logging.INFO)
smoke_test = "CI" in os.environ

import foraging_toolkit as ft


start_time = time.time()

hungry_sim = ft.Birds(
    grid_size=100, num_birds=3, num_frames=50, num_rewards=90, grab_range=3
)
hungry_sim()


hungry_sim = ft.add_hungry_birds(
    hungry_sim, num_hungry_birds=3, rewards_decay=0.3, visibility_range=6
)
end_time = time.time()

print("Generation time: ", end_time - start_time)
