# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
from typing import List
import random
import torch.nn as nn
import sys
from collections import defaultdict
import pickle


d=defaultdict(list)
for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
    d[k].append(v)

model = d.get('model')[0]

if model == 'dqn':
    from rh_marl_dqn.model import QNetwork
    from rh_marl_dqn.trainer import Trainer, Buffer
    from rh_marl_dqn._globals import GlobalVars
    from rh_marl_dqn.unity_integrate import separate_steps
elif model == 'atoc':
    from rh_marl_atoc.trainer import Trainer, Buffer
    from rh_marl_atoc.utils import Buffer, Trajectory, Experience
    from rh_marl_atoc._globals import GlobalVars
    from rh_marl_atoc.unity_integrate import separate_steps
    from rh_marl_atoc.model import ATOC, ATT  


with open ('output/'+GlobalVars.GAME_NAME+'/comm_no/training_cumulative_rewards', 'rb') as fp:
    comm_no_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_no/training_collisions_over_training_steps', 'rb') as fp:
    comm_no_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_no/training_coordinations_over_training_steps', 'rb') as fp:
    comm_no_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_no/training_episode_length_over_training_steps', 'rb') as fp:
    comm_no_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_no/training_episode_length_over_training_steps', 'rb') as fp:
    comm_no_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_no/training_goals_reached_over_training_steps', 'rb') as fp:
    comm_no_training_goals_reached_over_training_steps = pickle.load(fp)



with open ('output/'+GlobalVars.GAME_NAME+'/comm_atoc/training_cumulative_rewards', 'rb') as fp:
    comm_atoc_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_atoc/training_collisions_over_training_steps', 'rb') as fp:
    comm_atoc_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_atoc/training_coordinations_over_training_steps', 'rb') as fp:
    comm_atoc_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_atoc/training_episode_length_over_training_steps', 'rb') as fp:
    comm_atoc_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_atoc/training_episode_length_over_training_steps', 'rb') as fp:
    comm_atoc_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/comm_atoc/training_goals_reached_over_training_steps', 'rb') as fp:
    comm_atoc_training_goals_reached_over_training_steps = pickle.load(fp)
    
    
    
with open ('output/'+GlobalVars.GAME_NAME+'/comm_fully/training_cumulative_rewards', 'rb') as fp:
    comm_fully_training_cumulative_rewards = pickle.load(fp)[:200]
with open ('output/'+GlobalVars.GAME_NAME+'/comm_fully/training_collisions_over_training_steps', 'rb') as fp:
    comm_fully_training_collisions_over_training_steps = pickle.load(fp)[:200]
with open ('output/'+GlobalVars.GAME_NAME+'/comm_fully/training_coordinations_over_training_steps', 'rb') as fp:
    comm_fully_training_coordinations_over_training_steps = pickle.load(fp)[:200]
with open ('output/'+GlobalVars.GAME_NAME+'/comm_fully/training_episode_length_over_training_steps', 'rb') as fp:
    comm_fully_training_episode_length_over_training_steps = pickle.load(fp)[:200]
with open ('output/'+GlobalVars.GAME_NAME+'/comm_fully/training_episode_length_over_training_steps', 'rb') as fp:
    comm_fully_training_episode_length_over_training_steps = pickle.load(fp)[:200]
with open ('output/'+GlobalVars.GAME_NAME+'/comm_fully/training_goals_reached_over_training_steps', 'rb') as fp:
    comm_fully_training_goals_reached_over_training_steps = pickle.load(fp)[:200]
    
    



fig = plt.figure(1)

fig.suptitle('Game0 7x7 Multi Dynamic (comparison)')



plt.subplot(4, 1, 1)
plt.plot(np.array(comm_no_training_cumulative_rewards), 'r--', label='No communication')
plt.plot(np.array(comm_fully_training_cumulative_rewards), 'k--', label='Full communication')
plt.plot(np.array(comm_atoc_training_cumulative_rewards), 'c--', label='ATOC')
#plt.plot(np.array(testing_cumulative_rewards), 'g--',  label='testing')
plt.xlabel('training steps')
plt.ylabel('cumulative rewards') 
plt.legend()
plt.show()

plt.subplot(4, 1, 2)
plt.plot(np.array(comm_no_training_collisions_over_training_steps).mean(axis=1), 'r--', label='No communication', alpha=0.5)
plt.plot(np.array(comm_fully_training_collisions_over_training_steps).mean(axis=1), 'k--', label='Full communication')
plt.plot(np.array(comm_atoc_training_collisions_over_training_steps).mean(axis=1), 'c--', label='ATOC')
#plt.plot(np.array(testing_collisions_over_training_steps).mean(axis=1), 'g--', label='testing avg.')
plt.xlabel('training steps')
plt.ylabel('collisions (%)') 
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.subplot(4, 1, 3)
plt.plot(np.array(comm_no_training_coordinations_over_training_steps).mean(axis=1), 'r--', label='No communication')
plt.plot(np.array(comm_fully_training_coordinations_over_training_steps).mean(axis=1), 'k--', label='Full communication')
plt.plot(np.array(comm_atoc_training_coordinations_over_training_steps).mean(axis=1), 'c--', label='ATOC')
#plt.plot(np.array(testing_coordinations_over_training_steps).mean(axis=1), 'g--', label='testing avg.')
plt.xlabel('training steps')
plt.ylabel('communication (%)') 
plt.ylim(0, 1.1)
plt.legend()
plt.show()

plt.subplot(4, 1, 4)
plt.plot(np.array(comm_no_training_goals_reached_over_training_steps).mean(axis=1), 'r--', label='No communication')
plt.plot(np.array(comm_fully_training_goals_reached_over_training_steps).mean(axis=1), 'k--', label='Full communication')
plt.plot(np.array(comm_atoc_training_goals_reached_over_training_steps).mean(axis=1), 'c--', label='ATOC')
#plt.plot(np.array(training_coordinations_over_training_steps).mean(axis=1), 'r--', label='training avg.')
#plt.plot(np.array(testing_coordinations_over_training_steps).mean(axis=1), 'g--', label='testing avg.')
plt.xlabel('training steps')
plt.ylabel('goals reached') 
plt.legend()
plt.show()

