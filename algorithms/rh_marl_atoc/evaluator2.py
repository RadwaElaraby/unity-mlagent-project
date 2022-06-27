from mlagents_envs.environment import ActionTuple, BaseEnv
import numpy as np
import torch
from typing import Dict, List
import random
from ._globals import GlobalVars
from .unity_integrate import separate_steps
from .utils import Buffer, Trajectory, Experience
from .model import ATOC, ATT
from .action_to_state import action_to_new_pos, am_i_colliding_with_wall, am_i_colliding_with_another_agent, calculate_current_next_positions
from collections import deque
from statistics import median 
from .state_cnn import cnn_obs, ff_obs



class Evaluator:
  @staticmethod
  def generate_trajectories(env: BaseEnv, atoc: ATOC, att: ATT, buffer_size: int, epsilon: float):
    # Create an empty Buffer
    buffer: Buffer = []
    
    # Reset the environment
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]


    dict_backup_policy_how_many_times_counter: Dict[int, np.array] = {}
    dict_backup_policy_activated_max_steps: Dict[int, np.array] = {}
    dict_backup_policy_activated_counter: Dict[int, np.array] = {}
    dict_backup_policy_activated_random_counter: Dict[int, np.array] = {}
    dict_backup_policy_activated_actions_to_execute: Dict[int, np.ndarray] = {}
    dict_last_3_obs_from_agent: Dict[int, deque] = {}
    dict_last_3_action_from_agent: Dict[int, deque] = {}
    dict_last_reward_from_agent: Dict[int, np.ndarray] = {}

    dict_trajectories_from_agent: Dict[int, Trajectory] = {}
    dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
    dict_last_action_from_agent: Dict[int, np.ndarray] = {}
    dict_last_action_mask_from_agent: Dict[int, np.ndarray] = {}
    dict_last_comm_region_from_agent: Dict[int, np.ndarray] = {}
    dict_cumulative_reward_from_agent: Dict[int, np.ndarray] = {}
    cumulative_rewards: List[np.ndarray] = []
    dict_collisions_from_agent: Dict[int, np.ndarray] = {}    
    dict_wall_collisions_from_agent: Dict[int, np.ndarray] = {}    
    dict_agent_collisions_from_agent: Dict[int, np.ndarray] = {}    
    
    
    collisions: List[np.ndarray] = []
    wall_collisions: List[np.ndarray] = []
    agent_collisions: List[np.ndarray] = []
    dict_coordinations_from_agent: Dict[int, np.ndarray] = {}    
    coordinations: List[np.ndarray] = []
    dict_goals_reached_from_agent: Dict[int, np.ndarray] = {}    
    goals_reached: List[np.ndarray] = []            
    dict_epsiode_steps_from_agent: Dict[int, np.ndarray] = {}    
    epsiode_steps: List[np.ndarray] = []            


    steps_per_episodes = []
    steps_per_episode = 0
    
    
    while len(buffer) < buffer_size:  
      steps_per_episode += 1
                  
      act_decision_steps, act_terminal_steps = env.get_steps(behavior_name)      
      decision_steps, terminal_steps = separate_steps(act_decision_steps, act_terminal_steps)
      
      # For all Agents with a Terminal Step:
      for agent_id_terminated, _ in terminal_steps.items(): 
        if agent_id_terminated not in dict_last_action_from_agent:
            continue
        last_experience = Experience(
          obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
          action=dict_last_action_from_agent[agent_id_terminated].copy(),
          reward=terminal_steps[agent_id_terminated].group_reward + terminal_steps[agent_id_terminated].reward,
          next_obs=terminal_steps[agent_id_terminated].obs,
          done=not terminal_steps[agent_id_terminated].interrupted,
          comm_region=dict_last_comm_region_from_agent[agent_id_terminated].copy(),
          next_comm_region=terminal_steps[agent_id_terminated].comm_region,
          action_mask=None,
          next_action_mask=None
        )
        
        # Clear its last observation and action (Since the trajectory is over)
        dict_last_obs_from_agent.pop(agent_id_terminated)
        dict_last_action_from_agent.pop(agent_id_terminated)
        dict_last_comm_region_from_agent.pop(agent_id_terminated)
        dict_last_action_mask_from_agent.pop(agent_id_terminated)
        dict_last_3_obs_from_agent.pop(agent_id_terminated)
        dict_last_3_action_from_agent.pop(agent_id_terminated)
        dict_backup_policy_activated_counter.pop(agent_id_terminated)
        dict_backup_policy_activated_max_steps.pop(agent_id_terminated)
        dict_backup_policy_how_many_times_counter.pop(agent_id_terminated)
        dict_backup_policy_activated_random_counter.pop(agent_id_terminated)
        dict_backup_policy_activated_actions_to_execute.pop(agent_id_terminated)
        dict_last_reward_from_agent.pop(agent_id_terminated)
        
        cumulative_rewards.append(
            (dict_cumulative_reward_from_agent.pop(agent_id_terminated)
              + terminal_steps[agent_id_terminated].reward)
        )
        
        if GlobalVars.FINITE_EPISODE:              
            epsiode_steps.append(
                dict_epsiode_steps_from_agent.pop(agent_id_terminated) 
                + (dict_goals_reached_from_agent[agent_id_terminated] == 0)
            )
        else:
            epsiode_steps.append(dict_epsiode_steps_from_agent.pop(agent_id_terminated) + 1)
        
            
        
        if GlobalVars.FINITE_EPISODE: 
            collisions.append(
                  (dict_collisions_from_agent.pop(agent_id_terminated) 
                   + list(map(int, terminal_steps[agent_id_terminated].reward < GlobalVars.PENALTY_PER_STEP))
                  ) / np.array(epsiode_steps)[-1,:] 
            )
            wall_collisions.append(
                  (dict_wall_collisions_from_agent.pop(agent_id_terminated) 
                   + list(map(int, terminal_steps[agent_id_terminated].reward == GlobalVars.PENALTY_PER_STEP + GlobalVars.WALL_COLLISION_PENALTY_PER_STEP))
                  ) / np.array(epsiode_steps)[-1,:] 
            )
            agent_collisions.append(
                  (dict_agent_collisions_from_agent.pop(agent_id_terminated) 
                   + list(map(int, terminal_steps[agent_id_terminated].reward == GlobalVars.PENALTY_PER_STEP + GlobalVars.AGENT_COLLISION_PENALTY_PER_STEP))
                  ) / np.array(epsiode_steps)[-1,:] 
            )
        else:
            collisions.append(
                  (dict_collisions_from_agent.pop(agent_id_terminated) 
                   + list(map(int, terminal_steps[agent_id_terminated].reward < GlobalVars.PENALTY_PER_STEP))
                  ) / steps_per_episode
            )
            wall_collisions.append(
                  (dict_wall_collisions_from_agent.pop(agent_id_terminated) 
                   + list(map(int, terminal_steps[agent_id_terminated].reward == GlobalVars.PENALTY_PER_STEP + GlobalVars.WALL_COLLISION_PENALTY_PER_STEP))
                  ) / steps_per_episode
            )
            agent_collisions.append(
                  (dict_agent_collisions_from_agent.pop(agent_id_terminated) 
                   + list(map(int, terminal_steps[agent_id_terminated].reward == GlobalVars.PENALTY_PER_STEP + GlobalVars.AGENT_COLLISION_PENALTY_PER_STEP))
                  ) / steps_per_episode
            )            

        if GlobalVars.FINITE_EPISODE:   
            coordinations.append(
                    (
                        dict_coordinations_from_agent.pop(agent_id_terminated) 
                        + list(map(int, 
                                (terminal_steps[agent_id_terminated].comm_region.sum(axis=1) > 1) * (1 - dict_goals_reached_from_agent[agent_id_terminated])
                            )) 
                     ) / np.array(epsiode_steps)[-1,:] 
                ) 
        else:
            coordinations.append(
                  (dict_coordinations_from_agent.pop(agent_id_terminated) 
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
        
        
      
      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
          print('steps_per_episode', steps_per_episode)
          
      # For all Agents with a Decision Step:
      for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):

        # If the Agent does not have a Trajectory, create an empty one
        if agent_id_decisions not in dict_trajectories_from_agent:
          dict_trajectories_from_agent[agent_id_decisions] = []
          dict_cumulative_reward_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_wall_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_agent_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_coordinations_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_goals_reached_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))      
          dict_epsiode_steps_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))    
          dict_last_3_action_from_agent[agent_id_decisions] = deque(maxlen=GlobalVars.DEQUE_MAX_LENGTH)
          dict_last_3_obs_from_agent[agent_id_decisions] = deque(maxlen=GlobalVars.DEQUE_MAX_LENGTH)
          dict_backup_policy_activated_counter[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_backup_policy_activated_max_steps[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_backup_policy_how_many_times_counter[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_backup_policy_activated_random_counter[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_backup_policy_activated_actions_to_execute[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,8))
    

        # If the Agent requesting a decision has a "last observation"
        if agent_id_decisions in dict_last_obs_from_agent:

          if GlobalVars.IS_ACTION_SHARED or GlobalVars.IS_NEXT_OBS_SHARED:
              """
              evaluate action values without communication
              """
              no_communication = np.eye(GlobalVars.NUM_AGENTS, GlobalVars.NUM_AGENTS).astype(np.float32) 
              current_obs = np.array([v.obs for k,v in decision_steps.items()])

              atoc.eval()

    	                  
              if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
                  obs_walls = current_obs[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
                  obs_position = current_obs[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
                  obs_target = current_obs[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]

                  if GlobalVars.IS_CNN_ENCODER_ENABLED:
                      current_obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
                  else:                      
                      current_obs_for_cnn = ff_obs(current_obs, obs_walls, obs_position, obs_target)
        
                  actions_values_no_communication = (
                      atoc(torch.from_numpy(current_obs_for_cnn).cuda(), torch.from_numpy(no_communication).unsqueeze(dim=0).cuda())[0]
                  )
              else:    
                  actions_values_no_communication = (
                      atoc(torch.from_numpy(current_obs).cuda(), torch.from_numpy(no_communication).unsqueeze(dim=0).cuda())[0]
                  )
                    
              actions_values_no_communication = actions_values_no_communication.cpu()
              actions_no_communication = np.argmax(actions_values_no_communication.detach().numpy(), axis=1)
              atoc.train()
              
              if GlobalVars.IS_ACTION_SHARED:
                  """
                  update action slot in the observation of each agent
                  """
                  for agent_index in range(GlobalVars.NUM_AGENTS):
                      current_obs[0][agent_index][GlobalVars.ACTION_IN_STATE_INDEX : GlobalVars.ACTION_IN_STATE_INDEX+GlobalVars.OUTPUT_SIZE] += np.eye(GlobalVars.OUTPUT_SIZE)[actions_no_communication[agent_index]]
              else:
                  """
                  update next position slot in the observation of each agent
                  """
                  for agent_index in range(GlobalVars.NUM_AGENTS):
                      curr_walls = current_obs[0][agent_index][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                      curr_pos = current_obs[0][agent_index][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                      curr_action = actions_no_communication[agent_index]
                      new_pos = action_to_new_pos(curr_walls, curr_pos, curr_action)
                      new_pos = new_pos.flatten()
                      current_obs[0][agent_index][GlobalVars.NEXT_OBS_IN_STATE_INDEX : GlobalVars.NEXT_OBS_IN_STATE_INDEX+GlobalVars.GRID_SIZE] += new_pos
                    
              next_obs = current_obs[0]
          else:
              next_obs = decision_steps[agent_id_decisions].obs


          # Create an Experience from the last observation and the Decision Step
          exp = Experience(
            obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
            action=dict_last_action_from_agent[agent_id_decisions].copy(),
            reward=decision_steps[agent_id_decisions].group_reward + decision_steps[agent_id_decisions].reward,
            next_obs=next_obs,
            done=False,
            comm_region=dict_last_comm_region_from_agent[agent_id_decisions].copy(),
            next_comm_region=decision_steps[agent_id_decisions].comm_region,
            action_mask=dict_last_action_mask_from_agent[agent_id_decisions].copy(),
            next_action_mask=decision_steps[agent_id_decisions].action_mask
          )
  
          # Update the Trajectory of the Agent and its cumulative reward
          dict_trajectories_from_agent[agent_id_decisions].append(exp)
          dict_cumulative_reward_from_agent[agent_id_decisions] += (
              decision_steps[agent_id_decisions].reward
          )
          dict_collisions_from_agent[agent_id_decisions] += (
              list(map(int, decision_steps[agent_id_decisions].reward < GlobalVars.PENALTY_PER_STEP))
          ) 
          dict_wall_collisions_from_agent[agent_id_decisions] += (
              list(map(int, decision_steps[agent_id_decisions].reward == GlobalVars.PENALTY_PER_STEP + GlobalVars.WALL_COLLISION_PENALTY_PER_STEP))
          ) 
          dict_agent_collisions_from_agent[agent_id_decisions] += (
              list(map(int, decision_steps[agent_id_decisions].reward == GlobalVars.PENALTY_PER_STEP + GlobalVars.AGENT_COLLISION_PENALTY_PER_STEP))
          ) 
          
          if GlobalVars.FINITE_EPISODE:   
              dict_coordinations_from_agent[agent_id_decisions] += (
                list(map(int, 
                        (dict_last_comm_region_from_agent[agent_id_decisions][0].sum(axis=1) > 1) * (1 - dict_goals_reached_from_agent[agent_id_decisions]) # don't consider agents that already reached goal
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
         
          
        if GlobalVars.IS_ACTION_SHARED == False and GlobalVars.IS_NEXT_OBS_SHARED == False:
            # the state will be stored below to make sure the action slot is updated
            dict_last_obs_from_agent[agent_id_decisions] = (
                decision_steps[agent_id_decisions].obs
            )   
            d = decision_steps[agent_id_decisions].obs[:,GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
            r = []
            for idx in range(GlobalVars.NUM_AGENTS):
                r.append(np.where(d[idx]== 1)[0][0])
            dict_last_3_obs_from_agent[agent_id_decisions].append(r)

        dict_last_action_mask_from_agent[agent_id_decisions] = (
          decision_steps[agent_id_decisions].action_mask
        )
        
        

      all_obs = np.array([v.obs for k,v in decision_steps.items()])
      all_comm_regions = np.array([v.comm_region for k,v in decision_steps.items()])
    
      # if some agents have already terminated, add zeros for obs
      # they won't be used. it's just there to maintain shape of actions
      if (len(all_obs) == 0):
          all_obs = np.array([v.obs for k,v in terminal_steps.items()])
          all_comm_regions = np.array([v.comm_region for k,v in terminal_steps.items()])


      if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
          obs_walls = all_obs[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_position = all_obs[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_target = all_obs[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
        
          if GlobalVars.IS_CNN_ENCODER_ENABLED:
              all_obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
          else:                
              all_obs_for_cnn = ff_obs(all_obs, obs_walls, obs_position, obs_target)
      
          comm_values = np.array(att(torch.Tensor(all_obs_for_cnn).cuda(), torch.Tensor(all_comm_regions).cuda())[0].cpu().data)
      else:
          comm_values = np.array(att(torch.Tensor(all_obs).cuda(), torch.Tensor(all_comm_regions).cuda())[0].cpu().data)
      
        
      comm_values = comm_values.reshape(1, 4, 1)
      
      if (all_comm_regions.shape[0] > 1):
          raise 
      
      for agent_index in range(GlobalVars.NUM_AGENTS):
          # completely random
          if random.random() <= epsilon:
              all_comm_regions[0][agent_index] = all_comm_regions[0][agent_index] * (0 if random.random() < 0.5 else 1)  
          # attention unit decides whether communication is necessary
          else:
              #all_comm_regions[0][agent_index] = all_comm_regions[0][agent_index] * (0 if comm_values[0][agent_index][0] < GlobalVars.COMMUNICATION_THRESHOLD else 1)  
              all_comm_regions[0][agent_index] = all_comm_regions[0][agent_index] * (0 if comm_values[0][agent_index] < GlobalVars.COMMUNICATION_THRESHOLD else 1)  
                  
      all_comm_regions = all_comm_regions * GlobalVars.ALLOW_COMMUNICATION # + np.eye(GlobalVars.NUM_AGENTS).astype(np.float32) # << YOU NEED TO ADD EYE 
      np.fill_diagonal(all_comm_regions[0], 1)

     

      if GlobalVars.IS_ACTION_SHARED or GlobalVars.IS_NEXT_OBS_SHARED:
          """
          evaluate action values without communication
          """
          no_communication = np.eye(GlobalVars.NUM_AGENTS, GlobalVars.NUM_AGENTS).astype(np.float32) 
          atoc.eval()
          with torch.no_grad():
              if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
                  obs_walls = all_obs[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
                  obs_position = all_obs[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
                  obs_target = all_obs[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
                  
                  if GlobalVars.IS_CNN_ENCODER_ENABLED:
                      all_obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
                  else:
                      all_obs_for_cnn = ff_obs(all_obs, obs_walls, obs_position, obs_target)
                      
                  actions_values_no_communication = (
                      atoc(torch.from_numpy(all_obs_for_cnn).cuda(), torch.from_numpy(no_communication).unsqueeze(dim=0).cuda())[0]
                      )   
              else:  
                  actions_values_no_communication = (
                      atoc(torch.from_numpy(all_obs).cuda(), torch.from_numpy(no_communication).unsqueeze(dim=0).cuda())[0]
                  )   
                     
          actions_values_no_communication = actions_values_no_communication.cpu()
          actions_no_communication = np.argmax(actions_values_no_communication.detach().numpy(), axis=1)
          atoc.train()
          
          if GlobalVars.IS_ACTION_SHARED:
              """
              update action slot in the observation of each agent
              """
              for agent_index in range(GlobalVars.NUM_AGENTS):
                  all_obs[0][agent_index][GlobalVars.ACTION_IN_STATE_INDEX : GlobalVars.ACTION_IN_STATE_INDEX+GlobalVars.OUTPUT_SIZE] += np.eye(GlobalVars.OUTPUT_SIZE)[actions_no_communication[agent_index]]
          else:
              """
              update next position slot in the observation of each agent
              """
              for agent_index in range(GlobalVars.NUM_AGENTS):
                  curr_walls = all_obs[0][agent_index][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                  curr_pos = all_obs[0][agent_index][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                  curr_action = actions_no_communication[agent_index]
                  new_pos = action_to_new_pos(curr_walls, curr_pos, curr_action)
                  new_pos = new_pos.flatten()
                  #all_obs[0][agent_index][GlobalVars.NEXT_OBS_IN_STATE_INDEX : GlobalVars.NEXT_OBS_IN_STATE_INDEX+GlobalVars.GRID_SIZE] += new_pos
          
     
      # use trained policy
      if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
          obs_walls = all_obs[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_position = all_obs[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_target = all_obs[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
          
          if GlobalVars.IS_CNN_ENCODER_ENABLED:              
              all_obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
          else:
              all_obs_for_cnn = ff_obs(all_obs, obs_walls, obs_position, obs_target)
              
          
          
          actions_values = (
              atoc(torch.from_numpy(all_obs_for_cnn).cuda(), torch.from_numpy(all_comm_regions).cuda())[0]
          )            
          
      else:
          actions_values = (
              atoc(torch.from_numpy(all_obs).cuda(), torch.from_numpy(all_comm_regions).cuda())[0]
          )        
     
        


      actions_values = actions_values.cpu()

  
      if random.random() <= epsilon:
          # Add some noise with epsilon to the values
          actions_values = torch.from_numpy(np.random.randn(actions_values.shape[0], actions_values.shape[1]).astype(np.float32))
  
      # Pick the best action using argmax
      if GlobalVars.USE_STATIC_POLICY:
          # static policy 
          actions = np.argmax(actions_values.detach().numpy(), axis=1)
      else:
          # softmax policy
          action_list = np.array(list(range(GlobalVars.OUTPUT_SIZE))).astype(np.int64)
          actions = np.array([random.choices(action_list, weights=x) for x in actions_values])

      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:    
          print('actions from network', actions)

    
      """
      Agents that reached their goals always take action 0 
      """
      agents_terminated = np.zeros(GlobalVars.NUM_AGENTS, dtype=np.bool)
      for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
          for idx in range(GlobalVars.NUM_AGENTS):
              # detect the position of the agent
              curr_pos = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
              curr_tar = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_TARGET_MAP_INDEX : GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
          
              curr_pos_num = curr_pos.flatten()
              curr_pos_num = np.where(curr_pos_num == 1)[0][0]
              curr_tar_num = curr_tar.flatten()
              curr_tar_num = np.where(curr_tar_num == 1)[0][0]
              
              if curr_pos_num == curr_tar_num:
                  agents_terminated[idx] = True
                  actions[idx] = 0
            
      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
        print('agents_terminated', agents_terminated)
        print('actions + terminated', actions)
        
        
      dict_agent_collision_test_done = np.zeros(GlobalVars.NUM_AGENTS, dtype=np.bool)
      dict_agent_stuck_test_done = np.zeros(GlobalVars.NUM_AGENTS, dtype=np.bool)


      """
      Make sure the backup controller isn't enabled for too long for some reason
      """
      # Make sure backup controller isn't enabled for too long
      if GlobalVars.IS_STUCK_HANDLING_ENABLED or GlobalVars.IS_COLLISION_HANDLING_ENABLED:
          for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):     
              for idx in range(GlobalVars.NUM_AGENTS):
                  # counts number of steps during which backup controller was continuously enabled
                  if dict_backup_policy_activated_counter[agent_id_decisions][idx] > 1:
                      dict_backup_policy_activated_max_steps[agent_id_decisions][idx] += 1
                  else:
                      dict_backup_policy_activated_max_steps[agent_id_decisions][idx] = 0

                  if dict_backup_policy_activated_max_steps[agent_id_decisions][idx] > GlobalVars.BACKUP_CONTROLLER_MAX_CONSEQUENT_STEPS:
                      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                          print(idx, " backup controller was enabled for too long, enforce reusing network!") 
                      dict_backup_policy_activated_counter[agent_id_decisions][idx] = 0 
                      dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 0 
                      dict_agent_stuck_test_done[idx] = True


      """
      When the feature is enabled, Follow the backup controller for the agents that need it
      """
      # Apply backup controller if activated or wall avoidance if not activated
      if GlobalVars.IS_STUCK_HANDLING_ENABLED or GlobalVars.IS_COLLISION_HANDLING_ENABLED:
          for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
              
              for idx in range(GlobalVars.NUM_AGENTS):
                  curr_pos = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                  curr_walls = all_obs[agent_id_decisions][idx][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)

                  
                  """
                  Check if backup controller is enabled for the agent
                  """
                  # handle going back and forward differently because even backup controller will result in same going back and forward
                  if dict_backup_policy_activated_counter[agent_id_decisions][idx] <= 9 and dict_backup_policy_activated_counter[agent_id_decisions][idx] >= 2:
                    """
                    Execute the backtrace part of the backup controller
                    and if it is going to lead to a collision with wall, 
                        do nothing this timestep and proceed to random if part of the startegy next time step
                    """
                    if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                        print(idx, 'executing backup controller (default)')    
                      
                    # default handling of backup controller first steps
                    last_action = dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][int(dict_backup_policy_activated_counter[agent_id_decisions][idx]-2)] 
                      
                    
                    if last_action == 1:
                        actions[idx] = 2
                    elif last_action == 2:
                        actions[idx] = 1
                    elif last_action == 3:
                        actions[idx] = 4
                    elif last_action == 4:
                        actions[idx] = 3
                    else:
                        actions[idx] = 0
                    
                    if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                        print(idx, 'backup controller choose ', actions[idx])
                    
                    if am_i_colliding_with_wall(curr_walls, curr_pos, actions[idx]) :
                        if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                            print(idx, 'chosen action will lead to collision with a wall. stop and proceed to random part')
                        actions[idx] = 0
                        dict_backup_policy_activated_counter[agent_id_decisions][idx] = 1  
                    
                    else:
                        if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                            print(idx, 'safe proceed')
                        dict_backup_policy_activated_counter[agent_id_decisions][idx] -= 1
                        
                                      
                  # potential last 2 random actions
                  elif dict_backup_policy_activated_counter[agent_id_decisions][idx] == 1:
                      """
                      Execute the random part of the backup controller if enabled
                      Make sure the random action doesn't lead  to any wall collisions
                      """
                      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                          print(idx, 'executing backup controller (random part)')    
                      
                      if dict_backup_policy_activated_random_counter[agent_id_decisions][idx] > 0: 
                          last_action = dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][0] 
    
                          if last_action == 1:
                              random_actions = [0,1,3,4]
                          elif last_action == 2:
                              random_actions = [0,2,3,4]
                          elif last_action == 3:
                              random_actions = [0,1,2,3]
                          elif last_action == 4:
                              random_actions = [0,1,2,4]
                          else:
                              random_actions = [0,1,2,3,4]
                              
                          potential_actions = np.ones((GlobalVars.OUTPUT_SIZE,))
                          
                          potential_actions[list(set(range(GlobalVars.OUTPUT_SIZE)) - set(random_actions))] = 0
                          #p[actions_values[idx] <= min(actions_values[idx])] = 0
                          p = potential_actions / (potential_actions == 1).sum()
                          
                          actions[idx] = np.random.choice(list(range(GlobalVars.OUTPUT_SIZE)), 1, p=p)
                          while am_i_colliding_with_wall(curr_walls, curr_pos, actions[idx]):
                              potential_actions[actions[idx]] = 0
                              p = potential_actions / (potential_actions == 1).sum()
                              actions[idx] = np.random.choice(list(range(GlobalVars.OUTPUT_SIZE)), 1, p=p)

                          if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                              print(idx, 'backup controller choose ', actions[idx])
                              print(idx, 'safe proceed')
                              
                          dict_backup_policy_activated_random_counter[agent_id_decisions][idx] -= 1
                          
                      # noraml end of backup controller process    
                      else:
                          dict_backup_policy_activated_counter[agent_id_decisions][idx] = 0
                          dict_agent_stuck_test_done[idx] = True
                          

                  # if backup contronller is disabled, avoid walls 
                  else:
                      """
                      if backup isn't needed for that agent, adjust action if it is going to lead to wall collision
                      """                      
                      action_values_sorted = np.argsort(-actions_values.detach().numpy(), axis=1)

                      ## handle collisions with walls
                      ## if you are going to collide with a wall, choose the action with the next high Q value that doesn't lead to wall collisions
                      if GlobalVars.IS_COLLISION_HANDLING_ENABLED:
                          for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
              
                              if not agents_terminated[idx]: # dict_last_reward_from_agent[agent_id_decisions][idx] != 0:
                                  curr_pos = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                                  curr_walls = all_obs[agent_id_decisions][idx][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                                  curr_action = actions[idx]
                                  n = 1
                                  while am_i_colliding_with_wall(curr_walls, curr_pos, curr_action):
                                      actions[idx] = action_values_sorted[idx][n]
                                      curr_action = actions[idx]
                                      n = n + 1
        
                
        
      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
          print('actions after backup/wall_avoidance', actions)
        
        
      """
      At this point, all agents have chosen their actions either through policy or backup controller, 
      they also make sure they are not going to collide with a wall
      """
        
      """
      check if the backup tracer needs to be enabled next timestep
      """
      # if agents are going to collide, use backup controller
      if GlobalVars.IS_COLLISION_HANDLING_ENABLED:
          if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:              
              print('check collision...')
          for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):

              if agent_id_decisions in dict_last_3_obs_from_agent and len(dict_last_3_action_from_agent[agent_id_decisions]) >= 2:#GlobalVars.DEQUE_MAX_LENGTH:
                  
                  """
                  compute current and next positions for all agents
                  """
                  # detect current positions and supposedly next positions (based on action) not taking into account other agents
                  next_positions = []
                  current_positions = []
                  for idx in range(GlobalVars.NUM_AGENTS):
                      # detect the position of the agent
                      curr_pos = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                      
                      # if the agent is terminated
                      if agents_terminated[idx]:
                          curr_pos = curr_pos.flatten()
                          curr_pos = np.where(curr_pos == 1)[0][0]
                          next_positions.append(curr_pos)
                          current_positions.append(curr_pos)
                      else:
                          curr_walls = all_obs[agent_id_decisions][idx][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                          curr_action = actions[idx]
                          new_pos = action_to_new_pos(curr_walls, curr_pos, curr_action)
                          new_pos = new_pos.flatten()
                          next_positions.append(np.where(new_pos == 1)[0][0])
                          curr_pos = curr_pos.flatten()
                          current_positions.append(np.where(curr_pos == 1)[0][0])
                          
                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                      print('current_positions', current_positions)
                      print('next_positions', next_positions)
                  
                    
                  """
                  A loop to decide dict_agent_enable_backup_due_to_collision
                  dict_agent_enable_backup_due_to_collision will have a boolean number determining if a collision will happen with another agent 
                  based on the action the agent decides (after consulting backup controller if enabled for the agent)
                  """
                  dict_agent_enable_backup_due_to_collision = np.zeros(GlobalVars.NUM_AGENTS, dtype=np.bool)
                  for idx in range(GlobalVars.NUM_AGENTS):
                      
                        if dict_agent_collision_test_done[idx]:
                            continue
                               
                        """
                        check if other agents are planning to enter same position
                        """
                        # if > 1 agent is planning to move to the same position
                        who_else = [i for i, x in enumerate(next_positions) if x == next_positions[idx]]                    
                        #print(idx, who_else)
                        if len(who_else) > 1:
                            if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                print(who_else, ' are trying to enter same position')
                                
                            """
                            switch to backup if another agent is trying to enter your same position
                            if no one of them is terminated, choose one of them randomly to continue their path & for the others, backup
                            if some of them are terminated, backup the nonterminted ones
                            """
                            #print('--')
                            agents_cant_move = np.array(list(range(GlobalVars.NUM_AGENTS)))[agents_terminated]
                            agents_cant_move = np.intersect1d(agents_cant_move, who_else)
                            if len(agents_cant_move) > 0:
                                if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                    print(agents_cant_move, 'is already terminated')
                                agents_idx_switch_to_backup = list(set(who_else) - set(agents_cant_move))
                                if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                    print(agents_idx_switch_to_backup, ' must backup')
                                dict_agent_enable_backup_due_to_collision[agents_idx_switch_to_backup] = True
                                dict_agent_collision_test_done[who_else] = True
                                continue
                            else:
                                agent_idx_continue = random.choice(who_else)
                                if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                    print(agent_idx_continue, ' is selected to continue')                                
                                agents_idx_switch_to_backup = list(set(who_else) - set([agent_idx_continue]))
                                if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                    print(agents_idx_switch_to_backup, ' are selected to backup')
                                dict_agent_enable_backup_due_to_collision[agents_idx_switch_to_backup] = True
                                dict_agent_collision_test_done[who_else] = True
                                dict_agent_stuck_test_done[agent_idx_continue] = True # <<<<<<<<<<<<<<???????????????????????????????????????????????????????
                                continue
                        
                        """
                        check if another agent is trying to switch with your current agent
                        """
                        who_switch = [i for i, (f,t) in enumerate(zip(current_positions, next_positions)) 
                                      if current_positions[idx] != next_positions[idx] 
                                      and t == current_positions[idx] and f == next_positions[idx]]
                        #print(idx, who_switch)'
                        """
                        if so, backup both of them ????????
                        """
                        if len(who_switch) > 0:
                            both = who_switch + [idx]
                            if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:                                
                                print(both, ' are trying to switch and all have to backup') # ?????? why not just backup one of them?????
                            #print(idx, 'try to switch')
                            dict_agent_enable_backup_due_to_collision[both] = True
                            dict_agent_collision_test_done[both] = True
                            dict_agent_stuck_test_done[both] = True # <<<<<<<<<<<<<<???????????????????????????????????????????????????????
                            continue
                
                      
                  for idx in range(GlobalVars.NUM_AGENTS):
                      """
                      if the unit decided that the agent should use backup controller, pause for one time step, and prepare action buffers for them
                      """
                      if dict_agent_enable_backup_due_to_collision[idx]:#: and dict_last_reward_from_agent[agent_id_decisions][idx] != 0:
                          actions[idx] = 0
                          # if already executing backup, pause for one step
                          
                          """
                          it the agent was trying to continue a backup path, pasue for one timestep and continue afterwards your old path
                          """
                          if dict_backup_policy_activated_counter[agent_id_decisions][idx] > 1:
                              dict_backup_policy_activated_counter[agent_id_decisions][idx] += 1
                              
                              
                          else:
                              """
                              otherwise, prepare settings for backup controller
                              """
                              if len(dict_last_3_action_from_agent[agent_id_decisions]) >= 5: 
                                  if dict_backup_policy_how_many_times_counter[agent_id_decisions][idx] > 2:
                                      
                                      dict_backup_policy_activated_counter[agent_id_decisions][idx] = 9
                                      dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 2#2
                                      dict_backup_policy_how_many_times_counter[agent_id_decisions][idx] += 1
                                      
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][7] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-1][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][6] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-2][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][5] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-3][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][4] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-4][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][3] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-1][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][2] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-2][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][1] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-3][idx]  
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][0] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-4][idx] 
                                  else:
                                      dict_backup_policy_activated_counter[agent_id_decisions][idx] = 5
                                      dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 0
                                      dict_backup_policy_how_many_times_counter[agent_id_decisions][idx] += 1
                                      
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][3] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-1][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][2] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-2][idx] 
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][1] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-3][idx]  
                                      dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][0] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-4][idx] 
                                      
                                      
                                      # in case it was stopped for the last 4 timesteps already, go directly to random
                                      if dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][0:4].sum() == 0: # <<<<<<<<<<< Whyy indented????
                                          dict_backup_policy_activated_counter[agent_id_decisions][idx] = 1
                                          dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 2    
                                          
                                      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                          print(idx, ' was idle for the last 4 timesteps, switch directly to random')
                                      
                                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                      print('actions_to_reverse', dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx])
                              else:
                                  dict_backup_policy_activated_counter[agent_id_decisions][idx] = 4
                                  dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 0
                                  dict_backup_policy_how_many_times_counter[agent_id_decisions][idx] += 1
                                  
                                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                      print(idx, dict_last_3_action_from_agent[agent_id_decisions])
                                  
                                  dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][2] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-1][idx]  
                                  dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][1] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-2][idx] 
                                  dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][0] = dict_last_3_action_from_agent[agent_id_decisions][len(dict_last_3_action_from_agent[agent_id_decisions])-3][idx] 

                                  # in case it was stopped for the last 4 timesteps already, go directly to random
                                  if dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx][0:3].sum() == 0:
                                      dict_backup_policy_activated_counter[agent_id_decisions][idx] = 1
                                      dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 2    
                                    
                                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                      print(idx, ' was idle for the last 4 timesteps, switch directly to random')

                                  
                                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                                      print('actions_to_reverse', dict_backup_policy_activated_actions_to_execute[agent_id_decisions][idx])
                                            
         
      if GlobalVars.IS_STUCK_HANDLING_ENABLED:          
          # check if agents are stuck with same action and position for long time (3 timesteps in a row)
          for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
              
              
              if agent_id_decisions in dict_last_3_obs_from_agent and len(dict_last_3_action_from_agent[agent_id_decisions]) == GlobalVars.DEQUE_MAX_LENGTH:
                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                      print('checking stuck handling')
                  
                  # check if the agent in the same spot for ### latest steps
                  tt = np.array(list(dict_last_3_obs_from_agent[agent_id_decisions]))[GlobalVars.STUCK_SAME_THRESHOLD_INDEX:,:]
                  is_agent_stuck_same_position = np.all(tt == tt[0,:], axis=(0))
                  is_agent_stuck_same_position[agents_terminated] = False
                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                      print('is_agent_stuck_same_position', is_agent_stuck_same_position)
                  
                  # check if going back and forward
                  tt = np.array(list(dict_last_3_obs_from_agent[agent_id_decisions]))[GlobalVars.FOR_BACK_THRESHOLD_INDEX:,:]
                  tt = np.transpose(tt, axes=(1,0))
                  tt = list(map(np.unique, tt))
                  tt = list(map(len, tt))
                  is_agent_stuck_going_back_forward = np.array(tt) <= 2
                  is_agent_stuck_going_back_forward[agents_terminated] = False
                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                      print('is_agent_stuck_going_back_forward', is_agent_stuck_going_back_forward)
    
                  # check if going back and forward
                  tt = np.array(list(dict_last_3_obs_from_agent[agent_id_decisions]))[GlobalVars.FOR_BACK_THRESHOLD_INDEX2:,:]
                  tt = np.transpose(tt, axes=(1,0))
                  tt = list(map(np.unique, tt))
                  tt = list(map(len, tt))
                  is_agent_stuck_going_back_forward2 = np.array(tt) <= 3
                  is_agent_stuck_going_back_forward2[agents_terminated] = False
                  if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                      print('is_agent_stuck_going_back_forward2', is_agent_stuck_going_back_forward2)
    
                  
                  for idx in range(GlobalVars.NUM_AGENTS):
                      if dict_agent_stuck_test_done[idx]:
                          continue
                      
                      # TODO action is already become the same so going back doesn't make sense ???
                      #and dict_last_reward_from_agent[agent_id_decisions][idx] == -0.3
                      # and random.random() > 0.5: # and is_agent_stuck_same_action[idx]:
                     
                      if is_agent_stuck_same_position[idx] and not agents_terminated[idx] and dict_backup_policy_activated_counter[agent_id_decisions][idx] == 0:
                          if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                              print(idx, ' is stuck in the same position')
                          dict_backup_policy_activated_counter[agent_id_decisions][idx] = 1
                          dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 2
                          actions[idx] = 0
                         
                      elif is_agent_stuck_going_back_forward[idx] and not agents_terminated[idx] and dict_backup_policy_activated_counter[agent_id_decisions][idx] == 0:
                          if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                              print(idx, ' is going back and forward for some time')
                          dict_backup_policy_activated_counter[agent_id_decisions][idx] = 1
                          dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 3
                          actions[idx] = 0
                          
                      elif is_agent_stuck_going_back_forward2[idx] and not agents_terminated[idx] and dict_backup_policy_activated_counter[agent_id_decisions][idx] == 0:
                          if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                              print(idx, ' is going back and forward for a longer time')
                          dict_backup_policy_activated_counter[agent_id_decisions][idx] = 1
                          dict_backup_policy_activated_random_counter[agent_id_decisions][idx] = 4
                          actions[idx] = 0
                          
              
      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
          print('dict_backup_policy_activated_counter', dict_backup_policy_activated_counter)
          print('dict_backup_policy_activated_random_counter', dict_backup_policy_activated_random_counter)
          print('dict_backup_policy_how_many_times_counter', dict_backup_policy_how_many_times_counter)
          print('actions + agent collision check', actions)



      """
      final double check that the agent will not collide with a wall or an agent. if so, just stop!
      """
      # final ensurance that no collide is going to happen. otherwise, stop
      # may be useful becasue buffer is still empty or for any other reason
      if GlobalVars.IS_STUCK_HANDLING_ENABLED or GlobalVars.IS_COLLISION_HANDLING_ENABLED:
          for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
              if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                  print('final check for wall collisions!!!')
              for idx in range(GlobalVars.NUM_AGENTS):
                  curr_pos = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                  curr_walls = all_obs[agent_id_decisions][idx][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)

                  if am_i_colliding_with_wall(curr_walls, curr_pos, actions[idx]):
                    if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                        print(idx, 'final action will lead to collision with a wall. enforce stop!!!')
                    actions[idx] = 0
                    
              current_positions, next_positions = calculate_current_next_positions(all_obs, actions, agents_terminated, agent_id_decisions)
              
              if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                  print('final check for agent collisions!!!')

              for idx in range(GlobalVars.NUM_AGENTS):
                  curr_pos = all_obs[agent_id_decisions][idx][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
                  curr_walls = all_obs[agent_id_decisions][idx][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)


                  if am_i_colliding_with_another_agent(idx, current_positions, next_positions):
                    if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
                        print(idx, 'final action will lead to collision with another agent. enforce stop!!!!')
                    actions[idx] = 0

      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
          print('actions + final check', actions)
      
      #print(actions)
      actions.resize((GlobalVars.NUM_AGENTS, 1))
      
      


      # Store the action that was picked, it will be put in the trajectory later
      #for agent_index, agent_id in enumerate(decision_steps.agent_id):
      for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
        if GlobalVars.IS_ACTION_SHARED or GlobalVars.IS_NEXT_OBS_SHARED:
            # Store the observation (after the action slot is given)
            dict_last_obs_from_agent[agent_id_decisions] = (
                all_obs[0]
            )
            
            d = all_obs[0][:,GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
            r = []
            for idx in range(GlobalVars.NUM_AGENTS):
                r.append(np.where(d[idx,:]== 1)[0][0])
            dict_last_3_obs_from_agent[agent_id_decisions].append(r)

        # the stored action should be an array and not just a value
        dict_last_action_from_agent[agent_id_decisions] = actions
        dict_last_reward_from_agent[agent_id_decisions] = decision_steps[agent_id_decisions].reward

        if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:        
            print('decision_steps[agent_id_decisions].reward', decision_steps[agent_id_decisions].reward)
            
        dict_last_3_action_from_agent[agent_id_decisions].append(actions)
        # Store the communication region as the new "last observation"
        dict_last_comm_region_from_agent[agent_id_decisions] = (
          all_comm_regions
        )        
      

      # Set the actions in the environment
      # Unity Environments expect ActionTuple instances.
      actions = actions.reshape((1, GlobalVars.NUM_AGENTS * GlobalVars.ACTION_SIZE)) # 1 because centralized
      action_tuple = ActionTuple()
      action_tuple.add_discrete(actions)
      env.set_actions(behavior_name, action_tuple)

      # Perform a step in the simulation
      env.step()

      if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:      
          print('------------------------------------------')
      
      
    return buffer, \
            np.mean(cumulative_rewards, axis=0),  \
            np.mean(collisions, axis=0),  \
            np.mean(wall_collisions, axis=0),  \
            np.mean(agent_collisions, axis=0),  \
            np.array(coordinations).mean(axis=0),  \
            np.array(steps_per_episodes).mean(), \
            np.array(goals_reached).mean(axis=0), \
            np.array(epsiode_steps).mean(axis=0)