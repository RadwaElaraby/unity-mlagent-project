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



fig = plt.figure(1)

fig.suptitle('Game2 10x10 Dynamic + Walls (ATOC communication)') # Game1_9x9_Dynamic

if GlobalVars.FINITE_EPISODE:
    plt.subplot(5, 1, 1)
    #plt.plot(np.array(training_cumulative_rewards), 'k-')
    plt.plot(np.array(training_cumulative_rewards)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_cumulative_rewards)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_cumulative_rewards)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_cumulative_rewards)[:,3], 'c--', label='agent 4', alpha=0.3)
    plt.plot(np.array(training_cumulative_rewards).mean(axis=1), 'k-.', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('cumulative rewards') 
    plt.show()
    
    plt.subplot(5, 1, 2)
    plt.plot(np.array(training_episode_steps_over_training_steps)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_episode_steps_over_training_steps)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_episode_steps_over_training_steps)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_episode_steps_over_training_steps)[:,3], 'c--', label='agent 4', alpha=0.3)    
    plt.plot(np.array(training_episode_steps_over_training_steps).mean(axis=1), 'k-.', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('episode steps (%)') 
    #plt.legend()
    plt.show()
    
    """
    plt.subplot(6, 1, 3)
    plt.plot(np.array(training_episode_length_over_training_steps), 'k-')
    #plt.plot(np.array(testing_episode_length_over_training_steps), 'g--',  label='testing')
    plt.xlabel('training steps')
    plt.ylabel('episode length') 
    plt.show()
    """
    
    plt.subplot(5, 1, 3)
    plt.plot(np.array(training_collisions_over_training_steps)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_collisions_over_training_steps)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_collisions_over_training_steps)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_collisions_over_training_steps)[:,3], 'c--', label='agent 4', alpha=0.3)
    plt.plot(np.array(training_collisions_over_training_steps).mean(axis=1), 'k-.', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('collisions (%)') 
    #plt.ylim(0, 1)
    #plt.legend()
    plt.show()
    
    
    #££££££££££££££££££££££££
    plt.subplot(5, 1, 4)
    plt.plot(np.array(training_coordinations_over_training_steps)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_coordinations_over_training_steps)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_coordinations_over_training_steps)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_coordinations_over_training_steps)[:,3], 'c--', label='agent 4', alpha=0.3)    
    plt.plot(np.array(training_coordinations_over_training_steps).mean(axis=1), 'k-.', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('communication (%)') 
    #plt.legend()
    plt.show()
    
    
    plt.subplot(5, 1, 5)
    plt.plot(np.array(training_goals_reached_over_training_steps)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_goals_reached_over_training_steps)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_goals_reached_over_training_steps)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_goals_reached_over_training_steps)[:,3], 'c--', label='agent 4', alpha=0.3)    
    plt.plot(np.array(training_goals_reached_over_training_steps).mean(axis=1), 'k-.', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('reached goals (%)') 
    plt.legend()
    plt.show()    
    
    
    
else:
    
    plt.subplot(4, 1, 1)
    plt.plot(np.array(training_cumulative_rewards), 'k-', label='training')
    #plt.plot(np.array(testing_cumulative_rewards), 'g--',  label='testing')
    plt.xlabel('training steps')
    plt.ylabel('cumulative rewards') 
    plt.legend()
    plt.show()
    
    plt.subplot(4, 1, 2)
    plt.plot(np.array(training_collisions_over_training_steps)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_collisions_over_training_steps)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_collisions_over_training_steps)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_collisions_over_training_steps)[:,3], 'c--', label='agent 4', alpha=0.3)
    plt.plot(np.array(training_collisions_over_training_steps).mean(axis=1), 'k-', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('collisions (%)') 
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    
    plt.subplot(4, 1, 3)
    plt.plot(np.array(training_coordinations_over_training_steps)[:,0], 'r--', label='agent 1', alpha=0.3)
    plt.plot(np.array(training_coordinations_over_training_steps)[:,1], 'g--', label='agent 2.', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_coordinations_over_training_steps)[:,2], 'y--', label='agent 3.', alpha=0.3)
        plt.plot(np.array(training_coordinations_over_training_steps)[:,3], 'c--', label='agent 4.', alpha=0.3)
    plt.plot(np.array(training_coordinations_over_training_steps).mean(axis=1), 'k-', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('communication (%)') 
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()

    plt.subplot(4, 1, 4)
    plt.plot(np.array(training_goals_reached_over_training_steps)[:,0], 'r--', label='agent 1.', alpha=0.3)
    plt.plot(np.array(training_goals_reached_over_training_steps)[:,1], 'g--', label='agent 2', alpha=0.3)
    if GlobalVars.NUM_AGENTS > 2:
        plt.plot(np.array(training_goals_reached_over_training_steps)[:,2], 'y--', label='agent 3', alpha=0.3)
        plt.plot(np.array(training_goals_reached_over_training_steps)[:,3], 'c--', label='agent 4', alpha=0.3)
    plt.plot(np.array(training_goals_reached_over_training_steps).mean(axis=1), 'k-', label='avg.')
    plt.xlabel('training steps')
    plt.ylabel('goals reached') 
    plt.legend()
    plt.show()


