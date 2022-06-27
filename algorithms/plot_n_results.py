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





from rh_marl_atoc.trainer import Trainer, Buffer
from rh_marl_atoc.utils import Buffer, Trajectory, Experience
from rh_marl_atoc._globals import GlobalVars
from rh_marl_atoc.unity_integrate import separate_steps
from rh_marl_atoc.model import ATOC, ATT  


        

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_cumulative_rewards', 'rb') as fp:
    training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_collisions_over_training_steps', 'rb') as fp:
    training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_coordinations_over_training_steps', 'rb') as fp:
    training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_episode_length_over_training_steps', 'rb') as fp:
    training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_episode_steps_over_training_steps', 'rb') as fp:
    training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_goals_reached_over_training_steps', 'rb') as fp:
    training_goals_reached_over_training_steps = pickle.load(fp)


"""
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_cumulative_rewards', 'rb') as fp:
    testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_collisions_over_training_steps', 'rb') as fp:
    testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_coordinations_over_training_steps', 'rb') as fp:
    testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_length_over_training_steps', 'rb') as fp:
    testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_goals_reached_over_training_steps', 'rb') as fp:
    testing_goals_reached_over_training_steps = pickle.load(fp)
"""


algo_type = 'big network/longer training'

fig = plt.figure(1)

fig.suptitle('Game2 10x10 Dynamic (ATOC communication) - Comparison') # Game1_9x9_Dynamic

plt.subplot(5, 1, 1)
plt.plot(np.array(training_cumulative_rewards).mean(axis=1), 'g-.', label='avg. '+algo_type)
plt.xlabel('training steps')
plt.ylabel('cumulative rewards') 
plt.show()

plt.subplot(5, 1, 2) 
plt.plot(np.array(training_episode_steps_over_training_steps).mean(axis=1), 'g-.', label='avg. '+algo_type)
plt.xlabel('training steps')
plt.ylabel('episode length') 
#plt.legend()
plt.show()


plt.subplot(5, 1, 3)
plt.plot(np.array(training_collisions_over_training_steps).mean(axis=1), 'g-.', label='avg. '+algo_type)
plt.xlabel('training steps')
plt.ylabel('collisions (%)') 
#plt.ylim(0, 1)
#plt.legend()
plt.show()


#££££££££££££££££££££££££
plt.subplot(5, 1, 4)   
plt.plot(np.array(training_coordinations_over_training_steps).mean(axis=1), 'g-.', label='avg. '+algo_type)
plt.xlabel('training steps')
plt.ylabel('communication (%)') 
#plt.legend()
plt.show()


plt.subplot(5, 1, 5)
plt.plot(np.array(training_goals_reached_over_training_steps).mean(axis=1), 'g-.', label='avg. '+algo_type)
plt.xlabel('training steps')
plt.ylabel('reached goals (%)') 
plt.legend()
plt.show()    
    
    

