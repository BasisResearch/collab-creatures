# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:35:30 2023

Takes x and y locations of each bird at each point in time
and stores all the data in a data frame.

@author: admin
"""

import numpy as np
import pandas as pd

Nbirds = 5
Ntime = 10

bird_x = np.zeros([Nbirds, Ntime])
bird_y = np.zeros([Nbirds, Ntime])

for ti in range(Ntime):
    for bi in range(Nbirds):
        bird_x[bi, ti] = np.random.randn() # xloc 
        bird_y[bi, ti] = np.random.randn() # yloc


birds_all = []
for bi in range(Nbirds):
    single_bird = pd.DataFrame(
        {
            "x": bird_x[bi, :],
            "y": bird_y[bi, :],
            "time": range(1, Ntime + 1),
            "bird": bi + 1,
            "type": "random",
        }
    )

    birds_all.append(single_bird)
    
all_birds_data = pd.concat(birds_all)