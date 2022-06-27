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
GAME13 = 'Game2_10x10_Dynamic_Walls' # comm_atoc_eval_stuck_collision_v3_matrix_obs
GAME14 = "Game2_10x10_Dynamic_Walls" # comm_atoc_eval_stuck_collision_v3_numeric_vector_obs


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
            
        
        
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/testing_cumulative_rewards', 'rb') as fp:
    game11_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/testing_collisions_over_training_steps', 'rb') as fp:
    game11_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/testing_coordinations_over_training_steps', 'rb') as fp:
    game11_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/testing_episode_length_over_training_steps', 'rb') as fp:
    game11_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/testing_episode_steps_over_training_steps', 'rb') as fp:
    game11_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME11+'/comm_atoc_eval_stuck_collision_v3_binary_matrices_obs/testing_goals_reached_over_training_steps', 'rb') as fp:
    game11_testing_goals_reached_over_training_steps = pickle.load(fp)    
        
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/testing_cumulative_rewards', 'rb') as fp:
    game12_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/testing_collisions_over_training_steps', 'rb') as fp:
    game12_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/testing_coordinations_over_training_steps', 'rb') as fp:
    game12_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/testing_episode_length_over_training_steps', 'rb') as fp:
    game12_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/testing_episode_steps_over_training_steps', 'rb') as fp:
    game12_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME12+'/comm_atoc_eval_stuck_collision_v3/testing_goals_reached_over_training_steps', 'rb') as fp:
    game12_testing_goals_reached_over_training_steps = pickle.load(fp)       
    
    
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/testing_cumulative_rewards', 'rb') as fp:
    game13_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/testing_collisions_over_training_steps', 'rb') as fp:
    game13_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/testing_coordinations_over_training_steps', 'rb') as fp:
    game13_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/testing_episode_length_over_training_steps', 'rb') as fp:
    game13_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/testing_episode_steps_over_training_steps', 'rb') as fp:
    game13_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME13+'/comm_atoc_eval_stuck_collision_v3_matrix_obs/testing_goals_reached_over_training_steps', 'rb') as fp:
    game13_testing_goals_reached_over_training_steps = pickle.load(fp)    
             
    
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/testing_cumulative_rewards', 'rb') as fp:
    game14_testing_cumulative_rewards = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/testing_collisions_over_training_steps', 'rb') as fp:
    game14_testing_collisions_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/testing_coordinations_over_training_steps', 'rb') as fp:
    game14_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/testing_episode_length_over_training_steps', 'rb') as fp:
    game14_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/testing_episode_steps_over_training_steps', 'rb') as fp:
    game14_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('output/'+GAME14+'/comm_atoc_eval_stuck_collision_v3_numeric_vector_obs/testing_goals_reached_over_training_steps', 'rb') as fp:
    game14_testing_goals_reached_over_training_steps = pickle.load(fp)    
           
    


fig = plt.figure(1)

fig.suptitle('Impact of adding backup controller')




plt.subplot(1, 1, 1)
one = np.ones(1000)
one.fill(1)
plt.plot(one, 'k', label="1.0")
plt.plot(np.array(game6_testing_goals_reached_over_training_steps).mean(axis=1), 'b', label='latest 3 pos shared + backup cont.')
#plt.plot(np.array(game3_testing_goals_reached_over_training_steps).mean(axis=1), 'c--', label='latest obs shared', alpha=0.6)
#####plt.plot(np.array(game2_testing_cumulative_rewards).mean(axis=1), 'g--', label='action shared')
#####plt.plot(np.array(game4_testing_cumulative_rewards).mean(axis=1), 'c--', label='position shared')

#plt.plot(np.array(game1_testing_goals_reached_over_training_steps).mean(axis=1), 'y--', label='baseline', alpha=0.6)
#plt.plot(np.array(game5_testing_goals_reached_over_training_steps).mean(axis=1), 'g', label='baseline with backup cont.')
#plt.plot(np.array(game7_testing_goals_reached_over_training_steps).mean(axis=1), 'r', label='baseline + backup cont. 2')
#plt.plot(np.array(game8_testing_goals_reached_over_training_steps).mean(axis=1), '#7dd147', label='basline + CNN RMS + numeric + backup cont. 2')
#plt.plot(np.array(game9_testing_goals_reached_over_training_steps).mean(axis=1), 'b', label='baseline cnn (ids) with backup cont. 2')
plt.plot(np.array(game14_testing_goals_reached_over_training_steps).mean(axis=1), '#7982c9', label='baseline + numeric + backup cont. 3')
plt.plot(np.array(game12_testing_goals_reached_over_training_steps).mean(axis=1), '#9F0FD6', label='baseline + binary + backup cont. 3')
#plt.plot(np.array(game10_testing_goals_reached_over_training_steps).mean(axis=1), '#ff8e2e', label='basline + CNN Adam + numeric + backup cont. 2')
plt.plot(np.array(game13_testing_goals_reached_over_training_steps).mean(axis=1), '#d4177f', label='basline + CNN Adam + numeric + backup cont. 3')
plt.plot(np.array(game11_testing_goals_reached_over_training_steps).mean(axis=1), '#9c7722', label='basline + CNN Adam + binary + backup cont. 3')

plt.xlabel('timesteps (x 4000)')
plt.ylabel('task done probability') 
plt.legend()
plt.show()

plt.legend()
plt.show()

