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


from rh_marl_atoc.trainer import Trainer, Buffer
from rh_marl_atoc.utils import Buffer, Trajectory, Experience
from rh_marl_atoc._globals import GlobalVars
from rh_marl_atoc.unity_integrate import separate_steps
from rh_marl_atoc.model import ATOC, ATT  



GAME1 = 'Game2_10x10_Dynamic_Walls'
GAME2 = 'Game2_10x10_Dynamic_Walls_ActionShared'
GAME3 = 'Game2_10x10_Dynamic_Walls_3v'
GAME4 = 'Game2_10x10_Dynamic_Walls_NextObsShared'
GAME5 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision
GAME6 = 'Game2_10x10_Dynamic_Walls_3v' # comm_atoc_eval_stuck_collision


with open ('output/'+GAME1+'/comm_atoc_eval/training_cumulative_rewards', 'rb') as fp:
    game1_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/training_collisions_over_training_steps', 'rb') as fp:
    game1_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/training_coordinations_over_training_steps', 'rb') as fp:
    game1_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/training_episode_length_over_training_steps', 'rb') as fp:
    game1_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/training_episode_steps_over_training_steps', 'rb') as fp:
    game1_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/training_goals_reached_over_training_steps', 'rb') as fp:
    game1_training_goals_reached_over_training_steps = pickle.load(fp)
    
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/training_cumulative_rewards', 'rb') as fp:
    game5_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/training_collisions_over_training_steps', 'rb') as fp:
    game5_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/training_coordinations_over_training_steps', 'rb') as fp:
    game5_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/training_episode_length_over_training_steps', 'rb') as fp:
    game5_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/training_episode_steps_over_training_steps', 'rb') as fp:
    game5_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/training_goals_reached_over_training_steps', 'rb') as fp:
    game5_training_goals_reached_over_training_steps = pickle.load(fp)
    

 
with open ('output/'+GAME3+'/comm_atoc_eval/training_cumulative_rewards', 'rb') as fp:
    game3_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/training_collisions_over_training_steps', 'rb') as fp:
    game3_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/training_coordinations_over_training_steps', 'rb') as fp:
    game3_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/training_episode_length_over_training_steps', 'rb') as fp:
    game3_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/training_episode_steps_over_training_steps', 'rb') as fp:
    game3_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/training_goals_reached_over_training_steps', 'rb') as fp:
    game3_training_goals_reached_over_training_steps = pickle.load(fp)
    
    

with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/training_cumulative_rewards', 'rb') as fp:
    game6_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/training_collisions_over_training_steps', 'rb') as fp:
    game6_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/training_coordinations_over_training_steps', 'rb') as fp:
    game6_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/training_episode_length_over_training_steps', 'rb') as fp:
    game6_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/training_episode_steps_over_training_steps', 'rb') as fp:
    game6_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/training_goals_reached_over_training_steps', 'rb') as fp:
    game6_training_goals_reached_over_training_steps = pickle.load(fp)

fig = plt.figure(1)

fig.suptitle('Impact of adding backup controller')



plt.subplot(2, 1, 1)
plt.plot(np.array(game1_training_cumulative_rewards).mean(axis=1), 'r--', label='baseline')
plt.plot(np.array(game5_training_cumulative_rewards).mean(axis=1), 'k--', label='baseline with backup cont.')
#plt.plot(np.array(game2_training_cumulative_rewards).mean(axis=1), 'g--', label='action shared')
#plt.plot(np.array(game4_training_cumulative_rewards).mean(axis=1), 'c--', label='position shared')
plt.plot(np.array(game3_training_cumulative_rewards).mean(axis=1), 'b--', label='latest obs shared')
plt.plot(np.array(game6_training_cumulative_rewards).mean(axis=1), 'y--', label='latest obs shared with backup cont.')
plt.xlabel('training steps')
plt.ylabel('cumulative rewards') 
plt.legend()
plt.show()

plt.subplot(2, 1, 2)
ten = np.ones(750)
ten.fill(0.1)
plt.plot(ten, 'k', alpha=0.3)
ten = np.ones(750)
ten.fill(0.2)
plt.plot(ten,'k', alpha=0.3)
plt.plot(np.array(game1_training_collisions_over_training_steps).mean(axis=1), 'r--', label='baseline')
plt.plot(np.array(game5_training_collisions_over_training_steps).mean(axis=1), 'k--', label='baseline with backup cont.')
#plt.plot(np.array(game2_training_collisions_over_training_steps).mean(axis=1), 'g--', label='action shared')
#plt.plot(np.array(game4_training_collisions_over_training_steps).mean(axis=1), 'c--', label='position shared')
plt.plot(np.array(game3_training_collisions_over_training_steps).mean(axis=1), 'b--', label='latest obs shared')
plt.plot(np.array(game6_training_collisions_over_training_steps).mean(axis=1), 'y--', label='latest obs shared with backup cont.')
plt.xlabel('training steps')
plt.ylabel('collisions (%)') 
plt.legend()
plt.show()



