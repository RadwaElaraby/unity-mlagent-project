# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 21:19:46 2022

@author: radwa
"""
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
import numpy as np
import torch
from typing import Tuple
from math import floor
from typing import NamedTuple, List, Dict
from mlagents_envs.environment import ActionTuple, BaseEnv
from typing import Dict
import random
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import sys
from collections import deque
np.set_printoptions(threshold=sys.maxsize)


USE_PRE_SPECIFIED_ACTION = False
# TAKE CARE: THE SAME VAIRBLE EXISTS IN THE _globals.py!!
USE_STATIC_POLICY = False # argmax or softmax

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

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
    from rh_marl_atoc.evaluator2 import Evaluator
    from rh_marl_atoc.utils import Buffer, Trajectory, Experience
    from rh_marl_atoc._globals import GlobalVars
    from rh_marl_atoc.unity_integrate import separate_steps
    from rh_marl_atoc.model_cnn import ATOC, ATT  



if __name__ == "__main__": 
    
    try:
        env = UnityEnvironment('../game/MyPathFinder.exe', worker_id=GlobalVars.WORKER_ID, seed=1, side_channels=[])
        env.reset()
        behavior_name = list(env.behavior_specs)[0]    
        
        
        if model == 'dqn':
            print('model: dqn')
                
            # This is a non-blocking call that only loads the environment.
            qnet = QNetwork(input_size=GlobalVars.INPUT_SIZE, encoding_size=GlobalVars.GloblVarsENCODING_SIZE, output_size=GlobalVars.OUTPUT_SIZE)
            qnet.load_state_dict(torch.load("models/[GAME_ID]/qnet.pt"))
            
                    
            buffer: Buffer = []
    
            # Create a Mapping from AgentId to Trajectories. This will help us create
            # trajectories for each Agents
            dict_trajectories_from_agent: Dict[int, Trajectory] = {}
            # Create a Mapping from AgentId to the last observation of the Agent
            dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
            # Create a Mapping from AgentId to the last observation of the Agent
            dict_last_action_from_agent: Dict[int, np.ndarray] = {}
            # Create a Mapping from AgentId to cumulative reward (Only for reporting)
            dict_cumulative_reward_from_agent: Dict[int, float] = {}
            # Create a list to store the cumulative rewards obtained so far
            cumulative_rewards: List[float] = []
            
            last_experience_count = 0
            conflicts_count = 0
            
            for iteration in range(10):  # While not enough data in the buffer
                    
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                decision_steps, terminal_steps = separate_steps(decision_steps, terminal_steps)
                
                
                for agent_id_terminated in terminal_steps: #range(len(terminal_steps)):
                    if agent_id_terminated not in dict_last_action_from_agent:
                        continue
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                  
                      # Create its last experience (is last because the Agent terminated)
                    last_experience = Experience(
                      obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                      reward=terminal_steps[agent_id_terminated].group_reward + terminal_steps[agent_id_terminated].reward,
                      done=not terminal_steps[agent_id_terminated].interrupted,
                      action=dict_last_action_from_agent[agent_id_terminated].copy(),
                      next_obs=terminal_steps[agent_id_terminated].obs[0],
                    )
                    # Clear its last observation and action (Since the trajectory is over)
                    dict_last_obs_from_agent.pop(agent_id_terminated)
                    dict_last_action_from_agent.pop(agent_id_terminated)
                
                    # Report the cumulative reward
                    cumulative_reward = (
                      dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                      + terminal_steps[agent_id_terminated].group_reward 
                      + terminal_steps[agent_id_terminated].reward
                    )  
                    cumulative_rewards.append(cumulative_reward)
                    # Add the Trajectory and the last experience to the buffer
                    print('last_experience')
                    print(last_experience)
                    last_experience_count += 1
                    buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
                    buffer.append(last_experience)
            
                  
                  # For all Agents with a Decision Step:
                for agent_id_decisions in decision_steps: #range(len(decision_steps)):
                    # If the Agent does not have a Trajectory, create an empty one
                    if agent_id_decisions not in dict_trajectories_from_agent:
                      dict_trajectories_from_agent[agent_id_decisions] = []
                      dict_cumulative_reward_from_agent[agent_id_decisions] = 0
            
            
                    print('******************************************************')
                    # If the Agent requesting a decision has a "last observation"
                    if agent_id_decisions in dict_last_obs_from_agent:
                        
            
                      # Create an Experience from the last observation and the Decision Step
                      exp = Experience(
                        obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                        reward=decision_steps[agent_id_decisions].group_reward + decision_steps[agent_id_decisions].reward,
                        done=False,
                        action=dict_last_action_from_agent[agent_id_decisions].copy(),
                        next_obs=decision_steps[agent_id_decisions].obs[0],
                      )
                      print(exp)
                      # Update the Trajectory of the Agent and its cumulative reward
                      dict_trajectories_from_agent[agent_id_decisions].append(exp)
                      dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].group_reward + decision_steps[agent_id_decisions].reward
                      )
                      
                    # Store the observation as the new "last observation"
                    dict_last_obs_from_agent[agent_id_decisions] = (
                      decision_steps[agent_id_decisions].obs[0]
                    )
                    
            
            
            
                if (len(list(terminal_steps)) > 0):
                    print('terminal_steps group_reward reward')
                    for k, v in terminal_steps.items():
                        print(v.agent_id, v.group_reward, v.reward)
                        
                if (len(list(decision_steps)) > 0):
                    print('decision_steps group_reward reward')
                    for k, v in decision_steps.items():
                        print(v.agent_id, v.group_reward, v.reward)
            
            
                if (len(list(decision_steps)) > 0):
                    print('decision_steps')
                    
                    for k, v in decision_steps.items():
                        print(v.agent_id, v.obs)
            
                if (len(list(terminal_steps)) > 0):
                    print('terminal_steps')
                    for k, v in terminal_steps.items():
                        print(v.agent_id, v.obs)
             
                
                if USE_PRE_SPECIFIED_ACTION:        
                    pass
                else:
                    all_obs = np.array([v.obs[0][0:GlobalVars.INPUT_SIZE] for k,v in decision_steps.items()])
                    
                    if (len(all_obs) < GlobalVars.NUM_AGENTS):
                        adjusted_obs = np.zeros((GlobalVars.NUM_AGENTS, GlobalVars.INPUT_SIZE)).astype(np.float32)
                        for agent_id in range(GlobalVars.NUM_AGENTS):
                            if agent_id in decision_steps:
                                adjusted_obs[agent_id,] = all_obs[0,:]
                                np.delete(all_obs, (0), axis=0) # remove first row
                        all_obs = adjusted_obs  
                    
                    actions_values = (
                      qnet(torch.from_numpy(all_obs)).detach().numpy()
                    )
                    if USE_STATIC_POLICY:
                        # static policy 
                        actions = np.argmax(actions_values, axis=1)
                    else:
                        # random policy
                        action_list = np.array(list(range(GlobalVars.OUTPUT_SIZE))).astype(np.float32)
                        actions = np.array([random.choices(action_list, weights=x) for x in actions_values])
                    actions.resize((GlobalVars.NUM_AGENTS, 1))
               
                print('actions')
                print(actions, actions.shape)
            
                
                all_agent_ids = np.array([v.agent_id for k,v in decision_steps.items()])
                for agent_id in all_agent_ids:
                    dict_last_action_from_agent[agent_id] = actions[agent_id]
                    
                actions.resize((1, GlobalVars.NUM_AGENTS * GlobalVars.ACTION_SIZE))
                action_tuple = ActionTuple()
                action_tuple.add_discrete(actions)
                env.set_actions(behavior_name, action_tuple)
                
                
            
                print('----------------------------------------------')
                env.step()
         
            
            env.close()
            
        elif model == 'atoc':
            print('model: atoc')
           
            atoc = ATOC(n_agent=GlobalVars.NUM_AGENTS, num_inputs=GlobalVars.INPUT_SIZE, hidden_dim=GlobalVars.ENCODING_SIZE, num_actions=GlobalVars.OUTPUT_SIZE)
            atoc = atoc.cuda()

            att = ATT(din=GlobalVars.INPUT_SIZE).cuda()
            att = att.cuda()
            
            if USE_PRE_SPECIFIED_ACTION == False:
                atoc.load_state_dict(torch.load('./output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/atoc.pt'))     
                att.load_state_dict(torch.load('./output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/att.pt'))
                pass
            
            atoc.eval()
            att.eval()
            
            for n in range(1000):
                _, testing_rewards, testing_collisions, testing_wall_collisions, testing_agent_collisions, testing_coordinations, testing_length, testing_goals_reached, testing_episode_steps = Evaluator.generate_trajectories(env, atoc, att, buffer_size=1000, epsilon=0)
                
            env.close()          
        
    except Exception as e:
        env.close()
        raise(e)
        

def print_buffer(buffer):
    
    for b in buffer:
        print(b.obs[:,:-GlobalVars.NUM_AGENTS].reshape(GlobalVars.NUM_AGENTS, 3, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)[:,1,:].sum(axis=0))
        print(b.comm_region)
        print(b.action)
        print(b.reward)


#from collections import Counter
##rewards_groups = [e.reward for e in buffer]
#print(Counter(rewards_groups))