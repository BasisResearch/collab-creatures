# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:18:58 2023

@author: admin

"""

import numpy as np
import matplotlib.pyplot as plt 

# Get the x and y coordinates of a 2D grid with dimensions (edge_size x edge_size)
def create_2Dgrid(edge_size):
  stateID_arr = np.arange(0, edge_size**2, 1) # vector of stateIDs (Nstates x 1)
  x_flat = np.mod(stateID_arr, edge_size) # vector of x coordinates for each state 
  y_flat = np.reshape(np.reshape(x_flat,[edge_size,edge_size]).T, [edge_size**2])
  return x_flat, y_flat, stateID_arr

# Given a stateID and 2D grid dimensions return scalar x and y coordinates 
def loc1Dto2D(locs_arr, edge_size):
    '''
    Parameters
    ----------
    loc_arr : array of 1D location indices. Can also be a scalar
    edge_size : edge size of square grid world 
    Returns 
    -------
    x_arr, y_arr : array of x coordinates of each location and array of y coordinates 
    '''
    stateID_arr = np.arange(0, edge_size**2, 1)
    x_flat = np.mod(stateID_arr, edge_size)
    x_grid = np.reshape(x_flat,[edge_size,edge_size])
    y_flat = np.reshape(x_grid.T, [edge_size**2])
    x_arr = x_flat[locs_arr]
    y_arr = y_flat[locs_arr]
    return x_arr, y_arr


# Given a (Nstates, 1) vector of values at each state, display those values on a grid world
def plot_state_values_on_grid(state_values, edge_size):
  state_values_grid = np.reshape(state_values,[edge_size,edge_size])
  fig, ax = plt.subplots()
  im = plt.imshow(state_values_grid)
  return fig, ax, im