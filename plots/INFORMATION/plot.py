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

def disable_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

smoothin_window = 11
smoothing_ = 9


status = 'testing'


model = 'binary_ff'
with open ('./'+model+'/'+status+'_collisions_over_training_steps','rb') as fp:
    regular_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps','rb') as fp:
    regular_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards','rb') as fp:
    regular_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps','rb') as fp:
    regular_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps','rb') as fp:
    regular_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps','rb') as fp:
    regular_testing_goals_reached_over_training_steps = pickle.load(fp)


model = 'last3'
with open ('./'+model+'/'+status+'_collisions_over_training_steps','rb') as fp:
    last3_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps','rb') as fp:
    last3_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards','rb') as fp:
    last3_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps','rb') as fp:
    last3_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps','rb') as fp:
    last3_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps','rb') as fp:
    last3_testing_goals_reached_over_training_steps = pickle.load(fp)


model = 'nextact'
with open ('./'+model+'/'+status+'_collisions_over_training_steps','rb') as fp:
    nextact_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps','rb') as fp:
    nextact_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards','rb') as fp:
    nextact_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps','rb') as fp:
    nextact_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps','rb') as fp:
    nextact_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps','rb') as fp:
    nextact_testing_goals_reached_over_training_steps = pickle.load(fp)




model = 'nextloc'
with open ('./'+model+'/'+status+'_collisions_over_training_steps','rb') as fp:
    nextloc_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps','rb') as fp:
    nextloc_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards','rb') as fp:
    nextloc_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps','rb') as fp:
    nextloc_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps','rb') as fp:
    nextloc_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps','rb') as fp:
    nextloc_testing_goals_reached_over_training_steps = pickle.load(fp)



start = 999
ens = 1000


def subplot(id):
    fig, a =plt.subplots(1,id)

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    
    return fig
        
    


fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)    
plt.plot(np.array(regular_testing_cumulative_rewards).mean(axis=1), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextact_testing_cumulative_rewards).mean(axis=1), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextloc_testing_cumulative_rewards).mean(axis=1), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(np.array(last3_testing_cumulative_rewards).mean(axis=1), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('SharingInformationResults_CumulativeReward.png', dpi=300)


fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(savgol_filter(np.array(regular_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextact_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextloc_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(last3_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('SharingInformationResults_CumulativeReward1.png', dpi=300)


fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(np.array(regular_testing_collisions_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextact_testing_collisions_over_training_steps).mean(axis=1), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextloc_testing_collisions_over_training_steps).mean(axis=1), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(np.array(last3_testing_collisions_over_training_steps).mean(axis=1), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('SharingInformationResults_Collisions.png', dpi=300)




fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(savgol_filter(np.array(regular_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextact_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextloc_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(last3_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('SharingInformationResults_Collisions1.png', dpi=300)




fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(np.array(regular_testing_coordinations_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextact_testing_coordinations_over_training_steps).mean(axis=1), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextloc_testing_coordinations_over_training_steps).mean(axis=1), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(np.array(last3_testing_coordinations_over_training_steps).mean(axis=1), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('SharingInformationResults_Coordinations.png', dpi=300)



fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(savgol_filter(np.array(regular_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextact_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextloc_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(last3_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('SharingInformationResults_Coordinations1.png', dpi=300)


fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(np.array(regular_testing_episode_steps_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextact_testing_episode_steps_over_training_steps).mean(axis=1), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextloc_testing_episode_steps_over_training_steps).mean(axis=1), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(np.array(last3_testing_episode_steps_over_training_steps).mean(axis=1), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('SharingInformationResults_EpisodeLength.png', dpi=300)



fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(savgol_filter(np.array(regular_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextact_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextloc_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(last3_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('SharingInformationResults_EpisodeLength1.png', dpi=300)



fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(np.array(regular_testing_goals_reached_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextact_testing_goals_reached_over_training_steps).mean(axis=1), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextloc_testing_goals_reached_over_training_steps).mean(axis=1), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(np.array(last3_testing_goals_reached_over_training_steps).mean(axis=1), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('SharingInformationResults_GoalsReached.png', dpi=300)



fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(savgol_filter(np.array(regular_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextact_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextloc_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(last3_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('SharingInformationResults_GoalsReached1.png', dpi=300)




fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(np.array(regular_testing_episode_length_over_training_steps), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextact_testing_episode_length_over_training_steps), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(np.array(nextloc_testing_episode_length_over_training_steps), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(np.array(last3_testing_episode_length_over_training_steps), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('SharingInformationResults_EpisodeLengthPess.png', dpi=300)



fig = subplot(1)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.plot(savgol_filter(np.array(regular_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextact_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#FB9A61', label='BaseATOC-CNN-B-[FUT-ACT]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(nextloc_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#73AC61', label='BaseATOC-CNN-B-[FUT-LOC]', alpha=1, linewidth=0.8) 
plt.plot(savgol_filter(np.array(last3_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#5D6A89', label='BaseATOC-MLP-B-[PAST-3-LOC]', alpha=1, linewidth=0.8) 
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('SharingInformationResults_EpisodeLengthPess1.png', dpi=300)


