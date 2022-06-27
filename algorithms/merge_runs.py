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
from rh_marl_atoc._globals import GlobalVars

                

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/training_cumulative_rewards', 'rb') as fp:
    training_cumulative_rewards1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/training_cumulative_rewards', 'rb') as fp:
    training_cumulative_rewards2 = pickle.load(fp)
training_cumulative_rewards3 = training_cumulative_rewards1 + training_cumulative_rewards2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_cumulative_rewards', 'wb') as fp:
            pickle.dump(training_cumulative_rewards3, fp)


with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/testing_cumulative_rewards', 'rb') as fp:
    testing_cumulative_rewards1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/testing_cumulative_rewards', 'rb') as fp:
    testing_cumulative_rewards2 = pickle.load(fp)
testing_cumulative_rewards3 = testing_cumulative_rewards1 + testing_cumulative_rewards2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_cumulative_rewards', 'wb') as fp:
            pickle.dump(testing_cumulative_rewards3, fp)
    
    
    
    
    
    

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/training_collisions_over_training_steps', 'rb') as fp:
    training_collisions_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/training_collisions_over_training_steps', 'rb') as fp:
    training_collisions_over_training_steps2 = pickle.load(fp)
training_collisions_over_training_steps3 = training_collisions_over_training_steps1 + training_collisions_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_collisions_over_training_steps', 'wb') as fp:
            pickle.dump(training_collisions_over_training_steps3, fp)
    
    

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/testing_collisions_over_training_steps', 'rb') as fp:
    testing_collisions_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/testing_collisions_over_training_steps', 'rb') as fp:
    testing_collisions_over_training_steps2 = pickle.load(fp)    
testing_collisions_over_training_steps3 = testing_collisions_over_training_steps1 + testing_collisions_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_collisions_over_training_steps', 'wb') as fp:
            pickle.dump(testing_collisions_over_training_steps3, fp)
    
    
    
    
    
    
    
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/training_coordinations_over_training_steps', 'rb') as fp:
    training_coordinations_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/training_coordinations_over_training_steps', 'rb') as fp:
    training_coordinations_over_training_steps2 = pickle.load(fp)
training_coordinations_over_training_steps3 = training_coordinations_over_training_steps1 + training_coordinations_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_coordinations_over_training_steps', 'wb') as fp:
            pickle.dump(training_coordinations_over_training_steps3, fp)


with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/testing_coordinations_over_training_steps', 'rb') as fp:
    testing_coordinations_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/testing_coordinations_over_training_steps', 'rb') as fp:
    testing_coordinations_over_training_steps2 = pickle.load(fp)  
testing_coordinations_over_training_steps3 = testing_coordinations_over_training_steps1 + testing_coordinations_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_coordinations_over_training_steps', 'wb') as fp:
            pickle.dump(testing_coordinations_over_training_steps3, fp)
  
    
  
    
  
    




with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/training_episode_length_over_training_steps', 'rb') as fp:
    training_episode_length_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/training_episode_length_over_training_steps', 'rb') as fp:
    training_episode_length_over_training_steps2 = pickle.load(fp)
training_episode_length_over_training_steps3 = training_episode_length_over_training_steps1 + training_episode_length_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_episode_length_over_training_steps', 'wb') as fp:
            pickle.dump(training_episode_length_over_training_steps3, fp)

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/testing_episode_length_over_training_steps', 'rb') as fp:
    testing_episode_length_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/testing_episode_length_over_training_steps', 'rb') as fp:
    testing_episode_length_over_training_steps2 = pickle.load(fp)
testing_episode_length_over_training_steps3 = testing_episode_length_over_training_steps1 + testing_episode_length_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_length_over_training_steps', 'wb') as fp:
            pickle.dump(testing_episode_length_over_training_steps3, fp)



    



with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/training_goals_reached_over_training_steps', 'rb') as fp:
    training_goals_reached_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/training_goals_reached_over_training_steps', 'rb') as fp:
    training_goals_reached_over_training_steps2 = pickle.load(fp)
training_goals_reached_over_training_steps3 = training_goals_reached_over_training_steps1 + training_goals_reached_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_goals_reached_over_training_steps', 'wb') as fp:
            pickle.dump(training_goals_reached_over_training_steps3, fp)

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/testing_goals_reached_over_training_steps', 'rb') as fp:
    testing_goals_reached_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/testing_goals_reached_over_training_steps', 'rb') as fp:
    testing_goals_reached_over_training_steps2 = pickle.load(fp)
testing_goals_reached_over_training_steps3 = testing_goals_reached_over_training_steps1 + testing_goals_reached_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_goals_reached_over_training_steps', 'wb') as fp:
            pickle.dump(testing_goals_reached_over_training_steps3, fp)



    
  
    

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/training_episode_steps_over_training_steps', 'rb') as fp:
    training_episode_steps_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/training_episode_steps_over_training_steps', 'rb') as fp:
    training_episode_steps_over_training_steps2 = pickle.load(fp)
training_episode_steps_over_training_steps3 = training_episode_steps_over_training_steps1 + training_episode_steps_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_episode_steps_over_training_steps', 'wb') as fp:
            pickle.dump(training_episode_steps_over_training_steps3, fp)

with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 1/testing_episode_steps_over_training_steps', 'rb') as fp:
    testing_episode_steps_over_training_steps1 = pickle.load(fp)
with open ('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+' - 2/testing_episode_steps_over_training_steps', 'rb') as fp:
    testing_episode_steps_over_training_steps2 = pickle.load(fp)
testing_episode_steps_over_training_steps3 = testing_episode_steps_over_training_steps1 + testing_episode_steps_over_training_steps2
with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_steps_over_training_steps', 'wb') as fp:
            pickle.dump(testing_episode_steps_over_training_steps3, fp)