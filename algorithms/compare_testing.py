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
GAME7 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v2
GAME8 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v2_matrix_obs
GAME9 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v2_matrix_obs_ids
GAME10 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v2_matrix_obs_2


with open ('output/'+GAME1+'/comm_atoc_eval/testing_cumulative_rewards', 'rb') as fp:
    game1_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/testing_collisions_over_training_steps', 'rb') as fp:
    game1_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/testing_coordinations_over_training_steps', 'rb') as fp:
    game1_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/testing_episode_length_over_training_steps', 'rb') as fp:
    game1_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/testing_episode_steps_over_training_steps', 'rb') as fp:
    game1_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME1+'/comm_atoc_eval/testing_goals_reached_over_training_steps', 'rb') as fp:
    game1_testing_goals_reached_over_training_steps = pickle.load(fp)
    
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/testing_cumulative_rewards', 'rb') as fp:
    game5_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/testing_collisions_over_training_steps', 'rb') as fp:
    game5_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/testing_coordinations_over_training_steps', 'rb') as fp:
    game5_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/testing_episode_length_over_training_steps', 'rb') as fp:
    game5_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/testing_episode_steps_over_training_steps', 'rb') as fp:
    game5_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME5+'/comm_atoc_eval_stuck_collision/testing_goals_reached_over_training_steps', 'rb') as fp:
    game5_testing_goals_reached_over_training_steps = pickle.load(fp)
    
    
"""
with open ('output/'+GAME2+'/comm_atoc_eval/testing_cumulative_rewards', 'rb') as fp:
    game2_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/testing_collisions_over_training_steps', 'rb') as fp:
    game2_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/testing_coordinations_over_training_steps', 'rb') as fp:
    game2_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/testing_episode_length_over_training_steps', 'rb') as fp:
    game2_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/testing_episode_steps_over_training_steps', 'rb') as fp:
    game2_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/testing_goals_reached_over_training_steps', 'rb') as fp:
    game2_testing_goals_reached_over_training_steps = pickle.load(fp)


"""
 
with open ('output/'+GAME3+'/comm_atoc_eval/testing_cumulative_rewards', 'rb') as fp:
    game3_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/testing_collisions_over_training_steps', 'rb') as fp:
    game3_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/testing_coordinations_over_training_steps', 'rb') as fp:
    game3_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/testing_episode_length_over_training_steps', 'rb') as fp:
    game3_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/testing_episode_steps_over_training_steps', 'rb') as fp:
    game3_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME3+'/comm_atoc_eval/testing_goals_reached_over_training_steps', 'rb') as fp:
    game3_testing_goals_reached_over_training_steps = pickle.load(fp)
    
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/testing_cumulative_rewards', 'rb') as fp:
    game6_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/testing_collisions_over_training_steps', 'rb') as fp:
    game6_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/testing_coordinations_over_training_steps', 'rb') as fp:
    game6_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/testing_episode_length_over_training_steps', 'rb') as fp:
    game6_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/testing_episode_steps_over_training_steps', 'rb') as fp:
    game6_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME6+'/comm_atoc_eval_stuck_collision/testing_goals_reached_over_training_steps', 'rb') as fp:
    game6_testing_goals_reached_over_training_steps = pickle.load(fp)    
    


with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/testing_cumulative_rewards', 'rb') as fp:
    game7_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/testing_collisions_over_training_steps', 'rb') as fp:
    game7_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/testing_coordinations_over_training_steps', 'rb') as fp:
    game7_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/testing_episode_length_over_training_steps', 'rb') as fp:
    game7_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/testing_episode_steps_over_training_steps', 'rb') as fp:
    game7_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/testing_goals_reached_over_training_steps', 'rb') as fp:
    game7_testing_goals_reached_over_training_steps = pickle.load(fp)    
        
    
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/testing_cumulative_rewards', 'rb') as fp:
    game8_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/testing_collisions_over_training_steps', 'rb') as fp:
    game8_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/testing_coordinations_over_training_steps', 'rb') as fp:
    game8_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/testing_episode_length_over_training_steps', 'rb') as fp:
    game8_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/testing_episode_steps_over_training_steps', 'rb') as fp:
    game8_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/testing_goals_reached_over_training_steps', 'rb') as fp:
    game8_testing_goals_reached_over_training_steps = pickle.load(fp)    
        
    
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/testing_cumulative_rewards', 'rb') as fp:
    game9_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/testing_collisions_over_training_steps', 'rb') as fp:
    game9_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/testing_coordinations_over_training_steps', 'rb') as fp:
    game9_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/testing_episode_length_over_training_steps', 'rb') as fp:
    game9_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/testing_episode_steps_over_training_steps', 'rb') as fp:
    game9_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/testing_goals_reached_over_training_steps', 'rb') as fp:
    game9_testing_goals_reached_over_training_steps = pickle.load(fp)    
        
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/testing_cumulative_rewards', 'rb') as fp:
    game10_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/testing_collisions_over_training_steps', 'rb') as fp:
    game10_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/testing_coordinations_over_training_steps', 'rb') as fp:
    game10_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/testing_episode_length_over_training_steps', 'rb') as fp:
    game10_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/testing_episode_steps_over_training_steps', 'rb') as fp:
    game10_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/testing_goals_reached_over_training_steps', 'rb') as fp:
    game10_testing_goals_reached_over_training_steps = pickle.load(fp)    
        
    
"""

with open ('output/'+GAME4+'/comm_atoc_eval/testing_cumulative_rewards', 'rb') as fp:
    game4_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME4+'/comm_atoc_eval/testing_collisions_over_training_steps', 'rb') as fp:
    game4_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME4+'/comm_atoc_eval/testing_coordinations_over_training_steps', 'rb') as fp:
    game4_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME4+'/comm_atoc_eval/testing_episode_length_over_training_steps', 'rb') as fp:
    game4_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME4+'/comm_atoc_eval/testing_episode_steps_over_training_steps', 'rb') as fp:
    game4_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME4+'/comm_atoc_eval/testing_goals_reached_over_training_steps', 'rb') as fp:
    game4_testing_goals_reached_over_training_steps = pickle.load(fp)
    
"""

fig = plt.figure(1)

fig.suptitle('Impact of adding backup controller')



plt.subplot(2, 1, 1)
one = np.ones(750)
one.fill(1)
plt.plot(one, 'k')
one = np.ones(750)
one.fill(0.9)
plt.plot(one, 'k')
plt.plot(np.array(game6_testing_cumulative_rewards).mean(axis=1), 'c', label='latest 3 pos shared + backup cont.')
#plt.plot(np.array(game3_testing_cumulative_rewards).mean(axis=1), 'c--', label='latest obs shared', alpha=0.6)
####plt.plot(np.array(game2_testing_cumulative_rewards).mean(axis=1), 'g--', label='action shared')
####plt.plot(np.array(game4_testing_cumulative_rewards).mean(axis=1), 'c--', label='position shared')

plt.plot(np.array(game1_testing_cumulative_rewards).mean(axis=1), 'y--', label='baseline', alpha=0.6)
#plt.plot(np.array(game5_testing_cumulative_rewards).mean(axis=1), 'g', label='baseline + backup cont.')
plt.plot(np.array(game7_testing_cumulative_rewards).mean(axis=1), 'r', label='baseline + backup cont. 2')
plt.plot(np.array(game8_testing_cumulative_rewards).mean(axis=1), 'g', label='basline + CNN RMS + backup cont. 2')
#plt.plot(np.array(game9_testing_cumulative_rewards).mean(axis=1), 'b', label='basline + CNN (w ids) + backup cont. 2')
plt.plot(np.array(game10_testing_cumulative_rewards).mean(axis=1), '#831a1a', label='basline + CNN Adam + backup cont. 2')
plt.xlabel('timesteps (x 4000)')
plt.ylabel('cumulative rewards') 
plt.legend()
plt.show()

plt.subplot(2, 1, 2)
ten = np.ones(750)
ten.fill(0.1)
plt.plot(ten, 'k', alpha=0.3)
ten = np.ones(750)
ten.fill(0.05)
plt.plot(ten,'k', alpha=0.3)
ten = np.ones(750)
ten.fill(0.02)
plt.plot(ten,'k', alpha=0.3)
plt.plot(np.array(game6_testing_collisions_over_training_steps).mean(axis=1), 'c', label='latest 3 pos shared with backup cont.')
#plt.plot(np.array(game3_testing_collisions_over_training_steps).mean(axis=1), 'c--', label='latest obs shared', alpha=0.6)
####plt.plot(np.array(game2_testing_collisions_over_training_steps).mean(axis=1), 'g--', label='action shared')
####plt.plot(np.array(game4_testing_collisions_over_training_steps).mean(axis=1), 'c--', label='position shared')
plt.plot(np.array(game1_testing_collisions_over_training_steps).mean(axis=1), 'y--', label='baseline', alpha=0.6)
#plt.plot(np.array(game5_testing_collisions_over_training_steps).mean(axis=1), 'g', label='baseline + backup cont.')
plt.plot(np.array(game7_testing_collisions_over_training_steps).mean(axis=1), 'r', label='baseline + backup cont. 2')
plt.plot(np.array(game8_testing_collisions_over_training_steps).mean(axis=1), 'g', label='baseline + CNN RMS + backup cont. 2')
#plt.plot(np.array(game9_testing_collisions_over_training_steps).mean(axis=1), 'b', label='baseline cnn (ids) with backup cont. 2')
plt.plot(np.array(game10_testing_collisions_over_training_steps).mean(axis=1), '#831a1a', label='baseline + CNN Adam + backup cont. 2')

plt.xlabel('timesteps (x 4000)')
plt.ylabel('collisions (%)') 
plt.legend()
plt.show()




















"""
plt.subplot(4, 1, 3)
plt.plot(np.array(game1_testing_coordinations_over_training_steps).mean(axis=1), 'r--', label='motion shared')
plt.plot(np.array(game2_testing_coordinations_over_training_steps).mean(axis=1), 'k--', label='no motion shared')
#plt.plot(np.array(comm_atoc_testing_coordinations_over_training_steps).mean(axis=1), 'c--', label='ATOC')
#plt.plot(np.array(testing_coordinations_over_training_steps).mean(axis=1), 'g--', label='testing avg.')
plt.xlabel('testing steps')
plt.ylabel('communication (%)') 
plt.ylim(0, 1.1)
plt.legend()
plt.show()

plt.subplot(4, 1, 4)
plt.plot(np.array(game1_testing_episode_steps_over_training_steps).mean(axis=1), 'r--', label='motion shared')
plt.plot(np.array(game2_testing_episode_steps_over_training_steps).mean(axis=1), 'k--', label='no motion shared')
#plt.plot(np.array(comm_atoc_testing_goals_reached_over_training_steps).mean(axis=1), 'c--', label='ATOC')
#plt.plot(np.array(testing_coordinations_over_training_steps).mean(axis=1), 'r--', label='testing avg.')
#plt.plot(np.array(testing_coordinations_over_training_steps).mean(axis=1), 'g--', label='testing avg.')
plt.xlabel('testing steps')
plt.ylabel('episode length') 
plt.legend()
plt.show()
"""


"""
plt.subplot(6, 1, 3)
plt.plot(np.array(testing_collisions_over_training_steps)[:,0],'r', label='agent 1')
plt.plot(np.array(testing_collisions_over_training_steps)[:,1], 'g', label='agent 2')
plt.xlabel('testing steps')
plt.ylabel('collisions (%)') 
plt.legend()
plt.show()

plt.subplot(6, 1, 4)
plt.plot(np.array(testing_collisions_over_training_steps)[:,0],'r-', label='agent 1')
plt.plot(np.array(testing_collisions_over_training_steps)[:,1], 'g-', label='agent 2')
plt.xlabel('testing steps')
plt.ylabel('collisions (%)') 
plt.legend()
plt.show()
"""

"""



plt.figure(1)
#plt.plot(np.array(collisions_over_training_steps)[:,0], 'r')
#plt.plot(np.array(collisions_over_training_steps)[:,1], 'g')
#plt.plot(np.array(collisions_over_training_steps)[:,2], 'y')
plt.plot(np.array(collisions_over_training_steps).mean(axis=1), 'r')
#plt.plot(np.array(collisions_over_training_steps)[:,3], 'b')
plt.show()


plt.figure(2)

#plt.plot(np.array(coordinations_over_training_steps)[:,0], 'r')
##plt.plot(np.array(coordinations_over_training_steps)[:,1], 'g')
#plt.plot(np.array(coordinations_over_training_steps)[:,2], 'y')
#plt.plot(np.array(coordinations_over_training_steps)[:,3], 'b')
plt.plot(np.array(coordinations_over_training_steps).mean(axis=1), 'g')
plt.show()

 
plt.figure(3)
plt.plot(np.array(cumulative_rewards), 'b')
plt.show()


plt.figure(4)
plt.plot(np.array(collisions_over_training_steps).mean(axis=1), 'r')
plt.plot(np.array(coordinations_over_training_steps).mean(axis=1), 'g')
plt.plot(np.array(cumulative_rewards), 'b')
plt.show()
"""