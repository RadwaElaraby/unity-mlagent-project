"""
showing individual 
without backup
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

def disable_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


status = 'testing'
backup = ''


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



start = 999
ens = 1000

print('rewards')
print('binary ff new     ',
      round(np.array(binary_ff_testing_cumulative_rewards)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_cumulative_rewards)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_cumulative_rewards)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_cumulative_rewards)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_cumulative_rewards)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_cumulative_rewards)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_cumulative_rewards)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_cumulative_rewards)[start:ens].std(), 3))


print('length')
print('binary ff new     ',
      round(np.array(binary_ff_testing_episode_steps_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_episode_steps_over_training_steps)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_episode_steps_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_episode_steps_over_training_steps)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_episode_steps_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_episode_steps_over_training_steps)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_episode_steps_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_episode_steps_over_training_steps)[start:ens].std(), 3))



print('pess length')
print('binary ff new     ',
      round(np.array(binary_ff_testing_episode_length_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_episode_length_over_training_steps)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_episode_length_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_episode_length_over_training_steps)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_episode_length_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_episode_length_over_training_steps)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_episode_length_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_episode_length_over_training_steps)[start:ens].std(), 3))



print('reached')
print('binary ff new     ',
      round(np.array(binary_ff_testing_goals_reached_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_goals_reached_over_training_steps)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_goals_reached_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_goals_reached_over_training_steps)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_goals_reached_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_goals_reached_over_training_steps)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_goals_reached_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_goals_reached_over_training_steps)[start:ens].std(), 3))




print('collisions')
print('binary ff new     ',
      round(np.array(binary_ff_testing_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_collisions_over_training_steps)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_collisions_over_training_steps)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_collisions_over_training_steps)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_collisions_over_training_steps)[start:ens].std(), 3))


print('wall coll')
print('binary ff new     ',
      round(np.array(binary_ff_testing_wall_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_wall_collisions_over_training_steps)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_wall_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_wall_collisions_over_training_steps)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_wall_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_wall_collisions_over_training_steps)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_wall_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_wall_collisions_over_training_steps)[start:ens].std(), 3))


print('agent coll')
print('binary ff new     ',
      round(np.array(binary_ff_testing_agent_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_ff_testing_agent_collisions_over_training_steps)[start:ens].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_agent_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_cnn_testing_agent_collisions_over_training_steps)[start:ens].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_agent_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(binary_cnn_testing_agent_collisions_over_training_steps)[start:ens].std(), 3))
print('numeric ff       ', 
      round(np.array(numeric_ff_testing_agent_collisions_over_training_steps)[start:ens].mean(), 3), 
      round(np.array(numeric_ff_testing_agent_collisions_over_training_steps)[start:ens].std(), 3))



smoothin_window = 11
smoothing_ = 9
LABEL = ''

def plotting(id):
    fig, a =plt.subplots(1,id)
    plt.grid(color='k', linestyle='-', linewidth=0.25)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    return fig
        
    

fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_CumulativeReward.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_CumulativeReward1.png', dpi=300)


fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_Collisions.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_Collisions1.png', dpi=300)



fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_Coordinations.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_Coordinations1.png', dpi=300)


fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_EpisodeLength.png', dpi=300)



fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_EpisodeLength1.png', dpi=300)


fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_GoalsReached.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_GoalsReached1.png', dpi=300)


fig = plotting(1)
plt.plot(np.array(binary_ff_testing_agent_collisions_over_training_steps).mean(axis=1), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-B wall'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'k', label='BaseATOC-CNN-N wall'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'c', label='BaseATOC-CNN-B wall'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Miscoordinations (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_WallvsAgent.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Miscoordinations (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_MisCoordinations1.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Wall Collisions (%)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_WallCollisions1.png', dpi=300)


fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_episode_length_over_training_steps), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_length_over_training_steps), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_length_over_training_steps), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_length_over_training_steps), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_EpisodeLengthPess.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'y', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'r', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'b', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), 'g', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('FOUR/FourVariantsResults_EpisodeLengthPess1.png', dpi=300)
