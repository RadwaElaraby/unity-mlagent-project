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
GAME11 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v3_binary_matrices_obs_3
GAME12 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v3
GAME13 = "Game2_10x10_Dynamic_Walls" # comm_atoc_eval_stuck_collision_v3_matrix_obs
GAME14 = "Game2_10x10_Dynamic_Walls" # comm_atoc_eval_stuck_collision_v3_numeric_vector_obs


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
    
    
"""
with open ('output/'+GAME2+'/comm_atoc_eval/training_cumulative_rewards', 'rb') as fp:
    game2_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/training_collisions_over_training_steps', 'rb') as fp:
    game2_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/training_coordinations_over_training_steps', 'rb') as fp:
    game2_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/training_episode_length_over_training_steps', 'rb') as fp:
    game2_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/training_episode_steps_over_training_steps', 'rb') as fp:
    game2_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME2+'/comm_atoc_eval/training_goals_reached_over_training_steps', 'rb') as fp:
    game2_training_goals_reached_over_training_steps = pickle.load(fp)


"""
 
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
   
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/training_cumulative_rewards', 'rb') as fp:
    game7_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/training_collisions_over_training_steps', 'rb') as fp:
    game7_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/training_coordinations_over_training_steps', 'rb') as fp:
    game7_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/training_episode_length_over_training_steps', 'rb') as fp:
    game7_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/training_episode_steps_over_training_steps', 'rb') as fp:
    game7_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME7+'/comm_atoc_eval_stuck_collision_v2/training_goals_reached_over_training_steps', 'rb') as fp:
    game7_training_goals_reached_over_training_steps = pickle.load(fp)    
            
        
    
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/training_cumulative_rewards', 'rb') as fp:
    game8_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/training_collisions_over_training_steps', 'rb') as fp:
    game8_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/training_coordinations_over_training_steps', 'rb') as fp:
    game8_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/training_episode_length_over_training_steps', 'rb') as fp:
    game8_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/training_episode_steps_over_training_steps', 'rb') as fp:
    game8_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME8+'/comm_atoc_eval_stuck_collision_v2_matrix_obs/training_goals_reached_over_training_steps', 'rb') as fp:
    game8_training_goals_reached_over_training_steps = pickle.load(fp)    
        
    
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/training_cumulative_rewards', 'rb') as fp:
    game9_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/training_collisions_over_training_steps', 'rb') as fp:
    game9_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/training_coordinations_over_training_steps', 'rb') as fp:
    game9_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/training_episode_length_over_training_steps', 'rb') as fp:
    game9_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/training_episode_steps_over_training_steps', 'rb') as fp:
    game9_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME9+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_ids/training_goals_reached_over_training_steps', 'rb') as fp:
    game9_training_goals_reached_over_training_steps = pickle.load(fp)    
        
    
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/training_cumulative_rewards', 'rb') as fp:
    game10_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/training_collisions_over_training_steps', 'rb') as fp:
    game10_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/training_coordinations_over_training_steps', 'rb') as fp:
    game10_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/training_episode_length_over_training_steps', 'rb') as fp:
    game10_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/training_episode_steps_over_training_steps', 'rb') as fp:
    game10_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME10+'/comm_atoc_eval_stuck_collision_v2_matrix_obs_2/training_goals_reached_over_training_steps', 'rb') as fp:
    game10_training_goals_reached_over_training_steps = pickle.load(fp)    
            
        
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/training_cumulative_rewards', 'rb') as fp:
    game11_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/training_collisions_over_training_steps', 'rb') as fp:
    game11_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/training_coordinations_over_training_steps', 'rb') as fp:
    game11_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/training_episode_length_over_training_steps', 'rb') as fp:
    game11_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/training_episode_steps_over_training_steps', 'rb') as fp:
    game11_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/training_goals_reached_over_training_steps', 'rb') as fp:
    game11_training_goals_reached_over_training_steps = pickle.load(fp)    
        
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/training_cumulative_rewards', 'rb') as fp:
    game12_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/training_collisions_over_training_steps', 'rb') as fp:
    game12_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/training_coordinations_over_training_steps', 'rb') as fp:
    game12_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/training_episode_length_over_training_steps', 'rb') as fp:
    game12_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/training_episode_steps_over_training_steps', 'rb') as fp:
    game12_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/training_goals_reached_over_training_steps', 'rb') as fp:
    game12_training_goals_reached_over_training_steps = pickle.load(fp)    
        
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/training_cumulative_rewards', 'rb') as fp:
    game13_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/training_collisions_over_training_steps', 'rb') as fp:
    game13_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/training_coordinations_over_training_steps', 'rb') as fp:
    game13_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/training_episode_length_over_training_steps', 'rb') as fp:
    game13_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/training_episode_steps_over_training_steps', 'rb') as fp:
    game13_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/training_goals_reached_over_training_steps', 'rb') as fp:
    game13_training_goals_reached_over_training_steps = pickle.load(fp)    
        
    
        
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/training_cumulative_rewards', 'rb') as fp:
    game14_training_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/training_collisions_over_training_steps', 'rb') as fp:
    game14_training_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/training_coordinations_over_training_steps', 'rb') as fp:
    game14_training_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/training_episode_length_over_training_steps', 'rb') as fp:
    game14_training_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/training_episode_steps_over_training_steps', 'rb') as fp:
    game14_training_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/training_goals_reached_over_training_steps', 'rb') as fp:
    game14_training_goals_reached_over_training_steps = pickle.load(fp)    
           
    

fig = plt.figure(1)

fig.suptitle('Collisions progress over training')

plt.subplot(1, 1, 1)
ten = np.ones(1000)
ten.fill(0.1)
plt.plot(ten, 'k', label="0.1")
ten = np.ones(1000)
ten.fill(0.05)
plt.plot(ten,'k', label="0.05")
ten = np.ones(1000)
ten.fill(0.02)
plt.plot(ten,'k', label="0.02")
plt.plot(np.array(game6_training_collisions_over_training_steps).mean(axis=1), 'b', label='latest 3 pos shared')

#plt.plot(np.array(game3_training_collisions_over_training_steps).mean(axis=1), 'c--', label='latest obs shared', alpha=0.6)
####plt.plot(np.array(game2_training_collisions_over_training_steps).mean(axis=1), 'g--', label='action shared')
####plt.plot(np.array(game4_training_collisions_over_training_steps).mean(axis=1), 'c--', label='position shared')

#plt.plot(np.array(game1_training_collisions_over_training_steps).mean(axis=1), 'y--', label='baseline', alpha=0.6)
#plt.plot(np.array(game5_training_collisions_over_training_steps).mean(axis=1), 'g', label='baseline with backup cont.')
#plt.plot(np.array(game7_training_collisions_over_training_steps).mean(axis=1), 'r', label='baseline + backup cont. 2')
#plt.plot(np.array(game8_training_collisions_over_training_steps).mean(axis=1), '#7dd147', label='basline + CNN RMS + numeric + backup cont. 2')
#plt.plot(np.array(game9_training_collisions_over_training_steps).mean(axis=1), 'b', label='basline + CNN (w ids) + backup cont. 2')
plt.plot(np.array(game14_training_collisions_over_training_steps).mean(axis=1),  '#7982c9', label='baseline + numeric')
plt.plot(np.array(game12_training_collisions_over_training_steps).mean(axis=1),  '#9F0FD6', label='baseline + binary')
#plt.plot(np.array(game10_training_collisions_over_training_steps).mean(axis=1), '#ff8e2e',  label='basline + CNN Adam + numeric')
plt.plot(np.array(game13_training_collisions_over_training_steps).mean(axis=1),  '#d4177f', label='basline + CNN Adam + numeric')
plt.plot(np.array(game11_training_collisions_over_training_steps).mean(axis=1),  '#9c7722', label='basline + CNN Adam + binary')


plt.xlabel('timesteps (x 4000)')
plt.ylabel('collisions (%)') 
plt.legend()
plt.show()

