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
    from rh_marl_atoc.utils import Buffer, Trajectory, Experience
    from rh_marl_atoc._globals import GlobalVars
    from rh_marl_atoc.unity_integrate import separate_steps
    from rh_marl_atoc.model import ATOC, ATT  



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
                        
                      #print('decision_steps[agent_id_decisions].group_reward')
                      #print(decision_steps[agent_id_decisions].group_reward)
                      #print('decision_steps[agent_id_decisions].reward')
                      #print(decision_steps[agent_id_decisions].reward)
            
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
                #atoc.load_state_dict(torch.load("models/Game1_9x9_Infinite/atoc.pt"))
                #att.load_state_dict(torch.load("models/Game1_9x9_Infinite/att.pt"))
                atoc.load_state_dict(torch.load('./output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/atoc.pt'))     
                att.load_state_dict(torch.load('./output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/att.pt'))
    
            
            buffer: Buffer = []
    
            # Create a Mapping from AgentId to Trajectories. This will help us create
            # trajectories for each Agents
            dict_trajectories_from_agent: Dict[int, Trajectory] = {}
            # Create a Mapping from AgentId to the last observation of the Agent
            dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
            # Create a Mapping from AgentId to the last observation of the Agent
            dict_last_action_from_agent: Dict[int, np.ndarray] = {}
            # Create a Mapping from AgentId to cumulative reward (Only for reporting)
            dict_cumulative_reward_from_agent: Dict[int, np.ndarray] = {}
        
            dict_last_action_mask_from_agent: Dict[int, np.ndarray] = {}
                
            dict_last_comm_region_from_agent: Dict[int, np.ndarray] = {}
            
            # Create a Mapping from AgentId to #collisions (Only for reporting)
            dict_collisions_from_agent: Dict[int, np.ndarray] = {}    
            # Create a list to store the #collisions obtained so far
            collisions: List[np.ndarray] = []            
            
            # Create a Mapping from AgentId to #collisions (Only for reporting)
            dict_coordinations_from_agent: Dict[int, np.ndarray] = {}    
            # Create a list to store the #collisions obtained so far
            coordinations: List[np.ndarray] = []

            # Create a Mapping from AgentId to #collisions (Only for reporting)
            dict_goals_reached_from_agent: Dict[int, np.ndarray] = {}    
            # Create a list to store the #collisions obtained so far
            goals_reached: List[np.ndarray] = []            
            
            # Create a Mapping from AgentId to #episode_steps 
            dict_epsiode_steps_from_agent: Dict[int, np.ndarray] = {}    
            # Create a list to store the #episode_steps obtained so far
            epsiode_steps: List[np.ndarray] = []             
            
            
            experiences: Buffer = []
            optim_atoc = torch.optim.RMSprop(atoc.parameters(), lr = 0.0005)
    
                
            steps_per_episodes = []
                
            steps_per_episode = 0
            
            
            last_experience_count = 0
            conflicts_count = 0
            
            for iteration in range(1000):  # While not enough data in the buffer
                steps_per_episode += 1
                if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

                print('iteration', iteration)
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                decision_steps, terminal_steps = separate_steps(decision_steps, terminal_steps)
                
                # For all Agents with a Terminal Step:
                for agent_id_terminated, _ in terminal_steps.items(): #range(len(terminal_steps)):
                  
                  if agent_id_terminated not in dict_last_action_from_agent:
                      continue
    
                    # Create its last experience (is last because the Agent terminated)
                  last_experience = Experience(
                    obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
                    reward=terminal_steps[agent_id_terminated].group_reward + terminal_steps[agent_id_terminated].reward,
                    done=not terminal_steps[agent_id_terminated].interrupted,
                    action=dict_last_action_from_agent[agent_id_terminated].copy(),
                    next_obs=terminal_steps[agent_id_terminated].obs,
                    comm_region=dict_last_comm_region_from_agent[agent_id_terminated].copy(),
                    next_comm_region=terminal_steps[agent_id_terminated].comm_region,
                    action_mask=None,
                    next_action_mask=None
                  )
                  
                  if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:
                     #print('last_experience')
                     #print(last_experience)
                     t = dict_last_obs_from_agent[agent_id_decisions][:,:-GlobalVars.NUM_AGENTS].reshape(GlobalVars.NUM_AGENTS, 3, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)[:,1,:]
                     print(t.sum(axis=0))

                     #print(dict_last_obs_from_agent[agent_id_decisions][:,:-2].reshape(4, 2, 5, 5))
                     #print(dict_last_obs_from_agent[agent_id_decisions][:,-2:])
                     print('dict_collisions_from_agent', dict_collisions_from_agent)
                     print('dict_coordination_from_agent', dict_coordinations_from_agent)
                     print('dict_goals_reached_from_agent', dict_goals_reached_from_agent)
                     print('dict_epsiode_steps_from_agent', dict_epsiode_steps_from_agent)
                     print('dict_cumulative_reward_from_agent', dict_cumulative_reward_from_agent)


                  # Clear its last observation and action (Since the trajectory is over)
                  dict_last_obs_from_agent.pop(agent_id_terminated)
                  dict_last_action_from_agent.pop(agent_id_terminated)
                  dict_last_comm_region_from_agent.pop(agent_id_terminated)
                  dict_last_action_mask_from_agent.pop(agent_id_terminated)
          
                  # Report the cumulative reward
                  cumulative_reward = (
                    dict_cumulative_reward_from_agent.pop(agent_id_terminated)
                    + terminal_steps[agent_id_terminated].reward
                  )  
                  
                  if GlobalVars.FINITE_EPISODE:              
                        # only increment steps for agents that haven't reached goal  
                        epsiode_steps.append(
                            dict_epsiode_steps_from_agent.pop(agent_id_terminated) 
                            + (dict_goals_reached_from_agent[agent_id_terminated] == 0)
                        )
                  else:
                        epsiode_steps.append(dict_epsiode_steps_from_agent.pop(agent_id_terminated) + 1)
                    
                  
                  if GlobalVars.FINITE_EPISODE:              
                      collisions.append(
                            (dict_collisions_from_agent.pop(agent_id_terminated)
                            + list(map(int, terminal_steps[agent_id_terminated].reward < GlobalVars.PENALTY_PER_STEP)))
                            / np.array(epsiode_steps)[-1,:] 
                      )  
                  else:
                      collisions.append(
                            (dict_collisions_from_agent.pop(agent_id_terminated)
                            + list(map(int, terminal_steps[agent_id_terminated].reward < GlobalVars.PENALTY_PER_STEP)))
                            /steps_per_episode
                      )  
                  
                  if GlobalVars.FINITE_EPISODE:   
                        coordinations.append(
                                (
                                    dict_coordinations_from_agent.pop(agent_id_terminated) 
                                    + list(map(int, 
                                            # don't consider agents that already reached goal
                                            (terminal_steps[agent_id_terminated].comm_region.sum(axis=1) > 1) * (1 - dict_goals_reached_from_agent[agent_id_terminated])
                                        ))
                                 ) / np.array(epsiode_steps)[-1,:] 
                            ) 
                  else:
                        coordinations.append(
                              (
                                  dict_coordinations_from_agent.pop(agent_id_terminated) 
                                  + list(map(int, terminal_steps[agent_id_terminated].comm_region.sum(axis=1) > 1))
                               ) / steps_per_episode
                        ) 
                      
                  goals_reached.append(
                          dict_goals_reached_from_agent.pop(agent_id_terminated) 
                          + list(map(int, terminal_steps[agent_id_terminated].reward == GlobalVars.GOAL_REWARD))
                    )
       
                  steps_per_episodes.append(steps_per_episode)
                  
                  
                  # Add the Trajectory and the last experience to the buffer
                  buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
                  buffer.append(last_experience)
                  
                  steps_per_episode = 0
          
                  print(1)
                  print(iteration)
                  ##env.close()
                  ##sys.exit()
                  print('-----------------------------------------------')
                  
          
                for agent_id_decisions, _ in decision_steps.items():
                  # If the Agent does not have a Trajectory, create an empty one
                  if agent_id_decisions not in dict_trajectories_from_agent:
                    dict_trajectories_from_agent[agent_id_decisions] = []
                    dict_cumulative_reward_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
                    dict_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
                    dict_coordinations_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
                    dict_goals_reached_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
                    dict_epsiode_steps_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))     

                
                # For all Agents with a Decision Step:
                for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
                  #print('agent_id_decisions', agent_id_decisions)
                  if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:                  
                      print('******************************************************')
    
                  # If the Agent requesting a decision has a "last observation"
                  if agent_id_decisions in dict_last_obs_from_agent:
                      
                    #print('decision_steps[agent_id_decisions].group_reward')
                    #print(decision_steps[agent_id_decisions].group_reward)
                    #print('decision_steps[agent_id_decisions].reward')
                    #print(decision_steps[agent_id_decisions].reward)
          
                    # Create an Experience from the last observation and the Decision Step
                    exp = Experience(
                      obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
                      reward=decision_steps[agent_id_decisions].group_reward + decision_steps[agent_id_decisions].reward,
                      done=False,
                      action=dict_last_action_from_agent[agent_id_decisions].copy(),
                      next_obs=decision_steps[agent_id_decisions].obs,
                      comm_region=dict_last_comm_region_from_agent[agent_id_decisions].copy(),
                      next_comm_region=decision_steps[agent_id_decisions].comm_region,
                      action_mask=dict_last_action_mask_from_agent[agent_id_decisions].copy(),
                      next_action_mask=decision_steps[agent_id_decisions].action_mask
                    )
                    if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:                        
                        #print(dict_last_obs_from_agent[agent_id_decisions][:,:-GlobalVars.NUM_AGENTS].reshape(4, 3, 9, 9)[:,1,:])
                        t = dict_last_obs_from_agent[agent_id_decisions][:,:-GlobalVars.NUM_AGENTS].reshape(GlobalVars.NUM_AGENTS, 3, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)[:,1,:]
                        print(t.sum(axis=0))
                        print('last_comm_region')
                        print(dict_last_comm_region_from_agent[agent_id_decisions])
                        print('actions')
                        print(dict_last_action_from_agent[agent_id_decisions])
                        #print(dict_last_obs_from_agent[agent_id_decisions][:,-GlobalVars.NUM_AGENTS:])
                        pass
            
                    print('reward', exp.reward)
                    
                    # Update the Trajectory of the Agent and its cumulative reward
                    dict_trajectories_from_agent[agent_id_decisions].append(exp)
                    dict_cumulative_reward_from_agent[agent_id_decisions] += (
                        decision_steps[agent_id_decisions].reward
                    )
                    #print(decision_steps[agent_id_decisions].reward, sum(decision_steps[agent_id_decisions].reward < GlobalVars.PENALTY_PER_STEP))
                    dict_collisions_from_agent[agent_id_decisions] += (
                        list(map(int, decision_steps[agent_id_decisions].reward < GlobalVars.PENALTY_PER_STEP))
                    ) 
                    
                    if GlobalVars.FINITE_EPISODE:   
                        dict_coordinations_from_agent[agent_id_decisions] += (
                          list(
                              map(int, 
                                  # don't consider agents that already reached goal
                                  (dict_last_comm_region_from_agent[agent_id_decisions][0].sum(axis=1) > 1) * (1 - dict_goals_reached_from_agent[agent_id_decisions])
                              ))
                          ) 
                    else:
                        dict_coordinations_from_agent[agent_id_decisions] += (
                            list(map(int, dict_last_comm_region_from_agent[agent_id_decisions][0].sum(axis=1) > 1))
                        ) 
                    
                    if GlobalVars.FINITE_EPISODE:              
                        # only increment steps for agents that haven't reached goal
                        dict_epsiode_steps_from_agent[agent_id_decisions] += (dict_goals_reached_from_agent[agent_id_decisions] == 0)
                    else:
                        dict_epsiode_steps_from_agent[agent_id_decisions] += 1
                    
                    dict_goals_reached_from_agent[agent_id_decisions] += (
                        list(map(int, decision_steps[agent_id_decisions].reward == GlobalVars.GOAL_REWARD))
                    )   
                    
                    print('dict_last_comm_region_from_agent[agent_id_decisions]')
                    print(dict_last_comm_region_from_agent[agent_id_decisions])                    
                    print('dict_collisions_from_agent', dict_collisions_from_agent)
                    print('dict_coordinations_from_agent', dict_coordinations_from_agent)
                    print('dict_goals_reached_from_agent', dict_goals_reached_from_agent)
                    print('dict_epsiode_steps_from_agent', dict_epsiode_steps_from_agent)
                    print('dict_cumulative_reward_from_agent', dict_cumulative_reward_from_agent)

                    
                  # Store the observation as the new "last observation"
                  dict_last_obs_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].obs
                  )
                  
                  # Store the communication region as the new "last observation"
                  dict_last_action_mask_from_agent[agent_id_decisions] = (
                    decision_steps[agent_id_decisions].action_mask
                  )
                  
                  
                  
                if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:

                    if (len(list(terminal_steps)) > 0):
                        print('terminal_steps group_reward reward')
                        for k, v in terminal_steps.items():
                            print(v.agent_id, v.group_reward, v.reward)
                            
                    if (len(list(decision_steps)) > 0):
                        print('decision_steps group_reward reward')
                        for k, v in decision_steps.items():
                            print(v.agent_id, v.group_reward, v.reward)

                if USE_PRE_SPECIFIED_ACTION:        
                    pass
                else:
                    
                    all_obs = np.array([v.obs for k,v in decision_steps.items()])
                    all_comm_regions = np.array([v.comm_region for k,v in decision_steps.items()])
                    

                    
                    if (len(all_obs) == 0):
                      all_obs = np.array([v.obs for k,v in terminal_steps.items()])
                      all_comm_regions = np.array([v.comm_region for k,v in terminal_steps.items()])
                      if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:
                          print('all_comm_regions pre')                    
                          print(all_comm_regions)

                    threshold = GlobalVars.COMMUNICATION_THRESHOLD
                    # compute whether communication is necessary or not for each agent based on the attention unit
                    comm_values = np.array(att(torch.Tensor(np.array([all_obs])).cuda())[0].cpu().data)
                    
                    if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:
                        print('comm_values')
                        print(comm_values)

                    for agent_index in range(GlobalVars.NUM_AGENTS):
                        all_comm_regions[0][agent_index] = all_comm_regions[0][agent_index] * (0 if comm_values[0][agent_index][0] < threshold else 1)  
                                  
                    all_comm_regions = all_comm_regions * GlobalVars.ALLOW_COMMUNICATION #+ np.eye(GlobalVars.NUM_AGENTS).astype(np.float32)
                    np.fill_diagonal(all_comm_regions[0], 1)

                    
                    if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:
                        print('all_comm_regions after')
                        print(all_comm_regions)
    
                    actions_values = (
                      #q_net(torch.from_numpy(decision_steps.obs[0])).detach().numpy()
                          atoc(torch.from_numpy(all_obs).cuda(), torch.from_numpy(all_comm_regions).cuda())[0]
                      )
     
                    actions_values = actions_values.cpu()
    
                    if USE_STATIC_POLICY:
                        # static policy 
                        actions = np.argmax(actions_values.detach().numpy(), axis=1)
                    else:
                        if random.random() > 0.9:
                            # random policy
                            action_list = np.array(list(range(GlobalVars.OUTPUT_SIZE))).astype(np.float32)
                            actions = np.array([random.choices(action_list, weights=x) for x in actions_values])                            
                        else:
                            actions = np.argmax(actions_values.detach().numpy(), axis=1)
                    actions.resize((GlobalVars.NUM_AGENTS, 1))
               
            
                
                for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
                  # the stored action should be an array and not just a value
                  dict_last_action_from_agent[agent_id_decisions] = actions
               
                  # Store the communication region as the new "last observation"
                  dict_last_comm_region_from_agent[agent_id_decisions] = (
                      all_comm_regions
                  )        
              
                actions = actions.reshape((1, GlobalVars.NUM_AGENTS * GlobalVars.ACTION_SIZE))
                action_tuple = ActionTuple()
                action_tuple.add_discrete(actions)
                env.set_actions(behavior_name, action_tuple)
                
                
                if GlobalVars.SHOW_TEST_SCIPRT_DEBUG_MESSAGES:            
                    print('----------------------------------------------')
                    
                env.step()
                
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
        print('--------------------------------')


#from collections import Counter
##rewards_groups = [e.reward for e in buffer]
#print(Counter(rewards_groups))