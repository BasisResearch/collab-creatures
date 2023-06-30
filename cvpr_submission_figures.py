# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:47:57 2023

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import pickle
from pathlib import Path
import figures 
from importlib import reload
reload(figures)

plt.close('all')
figures.setup_fig()

# directory = 'C:/Users/admin/Dropbox/Code/Basis-code/collaborative-intelligence/simulated_data/'
directory = 'simulated_data/'


#%% Load data from different experimental conditions to make statistical comparisons

food_type = 'clustered'
# food_type = 'distributed'

if food_type == 'distributed': 
    filename1 = 'distr_ignorers'
    filename2 = 'distr_communicators'
    # filename1 = 'distr_ignorers_run2'
    # filename2 = 'distr_communicators_run2'
    
elif food_type == 'clustered':
    filename1 = 'clust_ignorers'
    filename2 = 'clust_communicators'

# load the dictionary
dict_ignorers = pickle.load(open(directory + filename1 + '.sav', 'rb'))
dict_communicators = pickle.load(open(directory + filename2 + '.sav', 'rb'))

# load the results
timetofood_ignorers = dict_ignorers['pop_mean_time_to_first_food']
timetofood_communicators = dict_communicators['pop_mean_time_to_first_food']
numbirdsfailed_ignorers = dict_ignorers['num_agents_failed_reach_food']
numbirdsfailed_communicators = dict_communicators['num_agents_failed_reach_food']

N_timesteps = dict_ignorers['N_timesteps']
N_agents = dict_ignorers['N_agents']

#%% Mann-Whitney statistical tests to test for statistical significance of differences
# between two types of agents (Ignorers vs Communicators.)

# Environmental condition 1: Distributed food statistics 
#   Performance metric 1: Time to food 
mwu_time = stats.mannwhitneyu(timetofood_ignorers, timetofood_communicators, method='auto', alternative='two-sided')

#   Performance metric 2: Number of birds that failed to reach food 
mwu_numfailed = stats.mannwhitneyu(numbirdsfailed_ignorers, numbirdsfailed_communicators, method='auto', alternative='two-sided')

# Environmental condition 2: Clustered food statistics 
#   Performance metric 1: Time to food 
#   Performance metric 2: Number of birds that failed to reach food 

#%% Medians

# 1) Distributed food, Time to food
median_timetofood_ignorers = np.median(timetofood_ignorers)
median_timetofood_communicators = np.median(timetofood_communicators)

# 2) Distributed food, Number of birds that failed
median_numbirdsfailed_ignorers = np.median(numbirdsfailed_ignorers)
median_numbirdsfailed_communicators = np.median(numbirdsfailed_communicators)

# # 3) Clustered food, Time to food
# median_clust_timetofood = np.median(array, axis=)

# # 4) Clustered food, Number of birds that failed
# median_clust_numbirdsfailed = np.median(array, axis=)

#%% Plots comparing Ignorers and Communicators for each condition

doLegend = True

# 1) Time to food
color_ignorers = 'blue'
color_communicators = 'red'
fig, ax = plt.subplots()
hist1data = ax.hist(timetofood_ignorers, bins=np.arange(N_timesteps + 2), color=color_ignorers, alpha=0.5, label='Ignorers')
hist2data = ax.hist(timetofood_communicators,  bins=np.arange(N_timesteps + 2),  color=color_communicators, alpha=0.5, label='Communicators')

ymin1, ymin2 = 0, 0
ymax1, ymax2 = np.max(hist1data[0]), np.max(hist2data[0]) 
ax.vlines(median_timetofood_ignorers, ymin1, ymax1, color=color_ignorers, linestyles='dashed')
ax.vlines(median_timetofood_communicators, ymin2, ymax2,  color=color_communicators, linestyles='dashed')
ax.set_xlabel('Time to food location')
ax.set_ylabel('Number of populations')
ax.set_xlim([0, N_timesteps+2 ])
ax.set_xticks(np.linspace(0, N_timesteps, 6).astype(int))
if doLegend:
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
          labelcolor=[color_ignorers, color_communicators], handlelength=0, handletextpad=0,)
# also try labelcolor='linecolor'
fig.tight_layout()

# 2) Number of birds that failed
fig, ax = plt.subplots()
hist1data = ax.hist(numbirdsfailed_ignorers, bins=np.arange(N_agents), color=color_ignorers, alpha=0.5, label='Ignorers')
hist2data = ax.hist(numbirdsfailed_communicators, bins=np.arange(N_agents),  color=color_communicators, alpha=0.5, label='Communicators')

ymin1, ymin2 = 0, 0
ymax1, ymax2 = np.max(hist1data[0]), np.max(hist2data[0]) 
ax.vlines(median_numbirdsfailed_ignorers, ymin1, ymax1, color=color_ignorers, linestyles='dashed')
ax.vlines(median_numbirdsfailed_communicators, ymin2, ymax2,  color=color_communicators, linestyles='dashed')
ax.set_xlabel('Number of failed foragers')
ax.set_ylabel('Number of populations')
ax.set_xlim([0, N_agents])
ax.set_xticks(np.linspace(0, N_agents, N_agents+1).astype(int))
if doLegend:
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
              labelcolor=[color_ignorers, color_communicators], handlelength=0, handletextpad=0,)
fig.tight_layout()




# 3) Clustered food, Time to food
# 4) Clustered food, Number of birds that failed