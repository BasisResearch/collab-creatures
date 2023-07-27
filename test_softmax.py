# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:39:32 2023

test softmax function

@author: admin
"""

# import gridworld_utils as util
import numpy as np
import matplotlib.pyplot as plt 
import figures 

figures.setup_fig()

def softmax(x, T=1):
    return np.exp(x / T) / np.sum(np.exp(x / T))

# def softmax(x, b=1):
#     beta = np.log(b)
#     return np.exp(x ) / np.sum(np.exp(beta * x))  

T1 = 0.01 # temperature
T2 = 1 
T3 = 10
x = np.linspace(0, 1, 100)
x = np.linspace(0, 0.3, 5)
y = softmax(x, T=T1)
y2 = softmax(x, T=T2)
y3 = softmax(x, T=T3)
fig, ax = plt.subplots(); 
ax.plot(x, y, '.-', label='T = ' + str(T1))
ax.plot(x, y2, '.-', color='k', label='T = ' + str(T2))
ax.plot(x, y3, '.-', label='T = ' + str(T3))
ax.legend()
ax.set_xlabel('Value of location i')
ax.set_ylabel('P(choose i)')
fig.tight_layout()