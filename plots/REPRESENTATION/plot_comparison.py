"""
compare back without backup
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
import pickle
from scipy.signal import savgol_filter


SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




smoothin_window = 11
smoothing_ = 9

status = 'testing'
backup = ''
LABEL = ''

model = 'binary_cnn'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    binary_cnn_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_wall_collisions_over_training_steps = pickle.load(fp)

model = 'binary_ff'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    binary_ff_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_wall_collisions_over_training_steps = pickle.load(fp)

model = 'numeric_ff'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    numeric_ff_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_wall_collisions_over_training_steps = pickle.load(fp)

model = 'numeric_cnn'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    numeric_cnn_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_wall_collisions_over_training_steps = pickle.load(fp)



fig = plt.figure(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCumulativeRewardbackup.png', dpi=300)


fig = plt.figure(2)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCumulativeReward1backup.png', dpi=300)


fig = plt.figure(3)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCollisionsbackup.png', dpi=300)

fig = plt.figure(4)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_),  'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCollisions1backup.png', dpi=300)

fig = plt.figure(5)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1),  'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCoordinationsbackup.png', dpi=300)


fig = plt.figure(6)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_),  'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCoordinations1backup.png', dpi=300)


fig = plt.figure(7)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1),  'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLengthbackup.png', dpi=300)

fig = plt.figure(8)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLength1backup.png', dpi=300)


fig = plt.figure(9)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsGoalsReachedbackup.png', dpi=300)

fig = plt.figure(10)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_),  'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsGoalsReached1backup.png', dpi=300)

fig = plt.figure(12)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_),  'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Miscoordinations (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsMisCoordinations1backup.png', dpi=300)

fig = plt.figure(13)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Wall Collisions (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsWallCollisions1backup.png', dpi=300)

fig = plt.figure(14)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(np.array(numeric_ff_testing_episode_length_over_training_steps), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_length_over_training_steps), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_length_over_training_steps), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_length_over_training_steps), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLengthPessbackup.png', dpi=300)

fig = plt.figure(15)
plt.grid(color='k', linestyle='-', linewidth=0.25)   
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLengthPess1backup.png', dpi=300)








status = 'testing'
backup = '_backup'
LABEL = '-[BACKUP]'

model = 'binary_cnn'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    binary_cnn_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_wall_collisions_over_training_steps = pickle.load(fp)


model = 'binary_ff'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    binary_ff_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_wall_collisions_over_training_steps = pickle.load(fp)


model = 'numeric_ff'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    numeric_ff_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_wall_collisions_over_training_steps = pickle.load(fp)

model = 'numeric_cnn'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    numeric_cnn_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_wall_collisions_over_training_steps = pickle.load(fp)



fig = plt.figure(1)
plt.plot(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1),  '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCumulativeRewardbackup.png', dpi=300)


fig = plt.figure(2)
plt.plot(savgol_filter(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCumulativeReward1backup.png', dpi=300)


fig = plt.figure(3)
plt.plot(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1),   '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCollisionsbackup.png', dpi=300)

fig = plt.figure(4)
plt.plot(savgol_filter(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCollisions1backup.png', dpi=300)


fig = plt.figure(5)
plt.plot(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1),'#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1),  '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCoordinationsbackup.png', dpi=300)


fig = plt.figure(6)
plt.plot(savgol_filter(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_),  '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsCoordinations1backup.png', dpi=300)


fig = plt.figure(7)
plt.plot(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLengthbackup.png', dpi=300)

fig = plt.figure(8)
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLength1backup.png', dpi=300)


fig = plt.figure(9)
plt.plot(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1),'#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsGoalsReachedbackup.png', dpi=300)


fig = plt.figure(10)
plt.plot(savgol_filter(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_),'#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsGoalsReached1backup.png', dpi=300)


fig = plt.figure(12)
plt.plot(savgol_filter(np.array(numeric_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_),'#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Miscoordinations (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsMisCoordinations1backup.png', dpi=300)


fig = plt.figure(13)
plt.plot(savgol_filter(np.array(numeric_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_),  '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_),'#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Wall Collisions (%)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsWallCollisions1backup.png', dpi=300)


fig = plt.figure(14)
plt.plot(np.array(numeric_ff_testing_episode_length_over_training_steps),  '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_length_over_training_steps), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_length_over_training_steps), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_length_over_training_steps),'#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLengthPessbackup.png', dpi=300)


fig = plt.figure(15)
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('COMPARE/AllCompareResultsEpisodeLengthPess1backup.png', dpi=300)

