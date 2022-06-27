from mlagents_envs.environment import ActionTuple, BaseEnv
import numpy as np
import torch
from typing import Dict, List
import random
from ._globals import GlobalVars
from .unity_integrate import separate_steps
from .utils import Buffer, Trajectory, Experience
from .model import ATOC, ATT2 as ATT
from .action_to_state import action_to_new_pos
from .state_cnn import cnn_obs, ff_obs

class Trainer:
  @staticmethod
  def generate_trajectories(env: BaseEnv, atoc: ATOC, att: ATT, buffer_size: int, epsilon: float):
    # Create an empty Buffer
    buffer: Buffer = []
    
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

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
        
        dict_last_obs_from_agent.pop(agent_id_terminated)
        dict_last_action_from_agent.pop(agent_id_terminated)
        dict_last_comm_region_from_agent.pop(agent_id_terminated)
        dict_last_action_mask_from_agent.pop(agent_id_terminated)

        # Report the cumulative reward
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

        buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
        buffer.append(last_experience)

        steps_per_episode = 0

        
      # For all Agents with a Decision Step:
      for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):

        if agent_id_decisions not in dict_trajectories_from_agent:
          dict_trajectories_from_agent[agent_id_decisions] = []
          dict_cumulative_reward_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_wall_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))	
          dict_agent_collisions_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_coordinations_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))
          dict_goals_reached_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))      
          dict_epsiode_steps_from_agent[agent_id_decisions] = np.zeros((GlobalVars.NUM_AGENTS,))      

        # If the Agent requesting a decision has a "last observation"
        if agent_id_decisions in dict_last_obs_from_agent:

          exp = Experience(
            obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
            action=dict_last_action_from_agent[agent_id_decisions].copy(),
            reward=decision_steps[agent_id_decisions].group_reward + decision_steps[agent_id_decisions].reward,
            next_obs=decision_steps[agent_id_decisions].obs,
            done=False,
            comm_region=dict_last_comm_region_from_agent[agent_id_decisions].copy(),
            next_comm_region=decision_steps[agent_id_decisions].comm_region,
            action_mask=dict_last_action_mask_from_agent[agent_id_decisions].copy(),
            next_action_mask=decision_steps[agent_id_decisions].action_mask
          )
  
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
            
        # Store the observation and the action mask as the new "last observation" and "action_mask"
        dict_last_obs_from_agent[agent_id_decisions] = (
          decision_steps[agent_id_decisions].obs
        )       
        dict_last_action_mask_from_agent[agent_id_decisions] = (
          decision_steps[agent_id_decisions].action_mask
        )
      

      all_obs = np.array([v.obs for k,v in decision_steps.items()])
      all_comm_regions = np.array([v.comm_region for k,v in decision_steps.items()])
    
      if (len(all_obs) == 0):
          all_obs = np.array([v.obs for k,v in terminal_steps.items()])
          all_comm_regions = np.array([v.comm_region for k,v in terminal_steps.items()])

      """
      if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
          obs_walls = all_obs[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_position = all_obs[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_target = all_obs[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
        
          if GlobalVars.IS_CNN_ENCODER_ENABLED:
              obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
          else:                
              obs_for_cnn = ff_obs(all_obs, obs_walls, obs_position, obs_target)
      
          comm_values = np.array(att(torch.Tensor(obs_for_cnn).cuda(), torch.Tensor(all_comm_regions).cuda())[0].cpu().data)
      else:
          comm_values = np.array(att(torch.Tensor(all_obs).cuda(), torch.Tensor(all_comm_regions).cuda())[0].cpu().data)
      """

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
              all_comm_regions[0][agent_index] = all_comm_regions[0][agent_index] * (0 if comm_values[0][agent_index] < GlobalVars.COMMUNICATION_THRESHOLD else 1)  
                  
      all_comm_regions = all_comm_regions * GlobalVars.ALLOW_COMMUNICATION # + np.eye(GlobalVars.NUM_AGENTS).astype(np.float32) # << YOU NEED TO ADD EYE 
      np.fill_diagonal(all_comm_regions[0], 1)

      
      if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
          obs_walls = all_obs[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_position = all_obs[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
          obs_target = all_obs[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
          
          if GlobalVars.IS_CNN_ENCODER_ENABLED:
              obs_for_cnn = cnn_obs(obs_walls,obs_position, obs_target)
          else:
              obs_for_cnn = ff_obs(all_obs, obs_walls,obs_position, obs_target)
              
          actions_values = (
              atoc(torch.from_numpy(obs_for_cnn).cuda(), torch.from_numpy(all_comm_regions).cuda())[0]
          )              
          
      else:
          actions_values = (
              atoc(torch.from_numpy(all_obs).cuda(), torch.from_numpy(all_comm_regions).cuda())[0]
          )        
      actions_values = actions_values.cpu()
      
    
      if random.random() <= epsilon:
          actions_values = torch.from_numpy(np.random.randn(actions_values.shape[0], actions_values.shape[1]).astype(np.float32))

      if GlobalVars.USE_STATIC_POLICY:
          actions = np.argmax(actions_values.detach().numpy(), axis=1)
      else:
          action_list = np.array(list(range(GlobalVars.OUTPUT_SIZE))).astype(np.int64)
          actions = np.array([random.choices(action_list, weights=x) for x in actions_values])
      
      actions.resize((GlobalVars.NUM_AGENTS, 1))
      

      for agent_id_decisions, _ in decision_steps.items(): #range(len(decision_steps)):
        dict_last_action_from_agent[agent_id_decisions] = actions
        dict_last_comm_region_from_agent[agent_id_decisions] = (
          all_comm_regions
        )        
      

      # Set the actions in the environment
      actions = actions.reshape((1, GlobalVars.NUM_AGENTS * GlobalVars.ACTION_SIZE)) # 1 because centralized
      action_tuple = ActionTuple()
      action_tuple.add_discrete(actions)
      env.set_actions(behavior_name, action_tuple)

      env.step()
      
      
    return buffer, \
            np.mean(cumulative_rewards, axis=0),  \
            np.mean(collisions, axis=0),   \
            np.mean(wall_collisions, axis=0),  \
            np.mean(agent_collisions, axis=0),  \
            np.array(coordinations).mean(axis=0),  \
            np.array(steps_per_episodes).mean(), \
            np.array(goals_reached).mean(axis=0), \
            np.array(epsiode_steps).mean(axis=0)




  @staticmethod
  def update_atoc_net(atoc: ATOC, atoc_target: ATOC, 
                      att: ATT, att_target: ATT, 
                      optim_atoc: torch.optim, optim_att: torch.optim,
                      criterion_att,
                      buffer: Buffer, action_size: int):
    BATCH_SIZE = 1000
    batch_size = min(len(buffer), BATCH_SIZE)
    
    random.shuffle(buffer)
    
    # Split the buffer into batches
    batches = [
      buffer[batch_size * start : batch_size * (start + 1)]
      for start in range(int(len(buffer) / batch_size))
    ]

    for _ in range(GlobalVars.NUM_EPOCH):
        
      for batch in batches:

        obs = torch.from_numpy(
            np.stack([ex.obs for ex in batch])
        ).cuda()
        obs_np = np.stack([ex.obs for ex in batch])
        action = torch.from_numpy(
            np.stack([ex.action for ex in batch])
        )
        reward = torch.from_numpy(
          np.array([ex.reward for ex in batch], dtype=np.float32)#.reshape(-1, 1)
        )        
        next_obs = torch.from_numpy(
            np.stack([ex.next_obs for ex in batch])
        ).cuda()
        next_obs_np = np.stack([ex.next_obs for ex in batch])
        comm_region = torch.from_numpy(
            np.stack([ex.comm_region for ex in batch])
        ).cuda()
        next_comm_region = torch.from_numpy(
            np.stack([ex.next_comm_region for ex in batch])
        ).cuda()
        done = torch.from_numpy(
          np.array([ex.done for ex in batch], dtype=np.float32)#.reshape(-1, 1)
        )
        
                
        """
        UPDATE ATTENTION UNIT PARAMETERS
        """

        
        self_comm = torch.Tensor(np.array([np.eye(GlobalVars.NUM_AGENTS)]*batch_size)).cuda()
  
        if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
            obs_walls = next_obs_np[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
            obs_position = next_obs_np[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
            obs_target = next_obs_np[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
            
            if GlobalVars.IS_CNN_ENCODER_ENABLED:
                next_obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
            else:                
                next_obs_for_cnn = ff_obs(next_obs_np, obs_walls, obs_position, obs_target)

            comm_values = att(torch.from_numpy(next_obs_for_cnn).cuda(), next_comm_region.squeeze(1))        

            target_comm_values = (atoc(torch.from_numpy(next_obs_for_cnn).cuda(), next_comm_region)).max(dim = 2)[0] \
                                - (atoc(torch.from_numpy(next_obs_for_cnn).cuda(), self_comm)).max(dim = 2)[0]


        else:    
            comm_values = att(next_obs, next_comm_region.squeeze(1))        

            target_comm_values = (atoc(next_obs, next_comm_region)).max(dim = 2)[0] - (atoc(next_obs, self_comm)).max(dim = 2)[0]
            
            
        target_comm_values = (target_comm_values - target_comm_values.mean()) / (target_comm_values.std()+0.000001) + 0.5
        target_comm_values = torch.clamp(target_comm_values, 0, 1).unsqueeze(-1).detach()
        
        loss_att = criterion_att(comm_values, target_comm_values)
        
        optim_att.zero_grad()
        loss_att.backward()
        optim_att.step()
        
        
        """
        UPDATE ATOC PARAMETERS
        """
        
        if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
            obs_walls = obs_np[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
            obs_position = obs_np[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
            obs_target = obs_np[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
            
            if GlobalVars.IS_CNN_ENCODER_ENABLED:
                obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
            else:
                obs_for_cnn = ff_obs(obs_np, obs_walls, obs_position, obs_target)
                
            q_values = atoc(torch.from_numpy(obs_for_cnn).cuda(), comm_region)
        else:    
            q_values = atoc(obs, comm_region)
            
        expected_q = np.array(q_values.cpu().data)
        next_comm_region = next_comm_region * GlobalVars.ALLOW_COMMUNICATION
        next_comm_region[0].fill_diagonal_(1)
        
        
        if GlobalVars.IS_INPUT_PREPROCESSING_ENABLED:
            obs_walls = next_obs_np[:,:,GlobalVars.WALL_MAP_INDEX:GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE]
            obs_position = next_obs_np[:,:,GlobalVars.CURRENT_POSITION_MAP_INDEX:GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE]
            obs_target = next_obs_np[:,:,GlobalVars.CURRENT_TARGET_MAP_INDEX:GlobalVars.CURRENT_TARGET_MAP_INDEX+GlobalVars.GRID_SIZE]
            
            if GlobalVars.IS_CNN_ENCODER_ENABLED:
                next_obs_for_cnn = cnn_obs(obs_walls, obs_position, obs_target)
            else:
                next_obs_for_cnn = ff_obs(next_obs_np, obs_walls, obs_position, obs_target)

            target_q_values = atoc_target(torch.from_numpy(next_obs_for_cnn).cuda(), next_comm_region)
        else:    
            target_q_values = atoc_target(next_obs, next_comm_region)
            
            
        target_q_values = target_q_values.max(dim = 2)[0]
        target_q_values = np.array(target_q_values.cpu().data)
        for j in range(batch_size):
            for i in range(GlobalVars.NUM_AGENTS):
                expected_q[j][i][action[j][i]] = reward[j][i] + (1-done[j])*GlobalVars.GAMMA*target_q_values[j][i]
		
        loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
        
        optim_atoc.zero_grad()
        loss.backward()
        optim_atoc.step()
        
            
      with torch.no_grad():
          for p, p_targ in zip(atoc.parameters(), atoc_target.parameters()):
              p_targ.data.mul_(GlobalVars.TAU)
              p_targ.data.add_((1 - GlobalVars.TAU) * p.data)
          for p, p_targ in zip(att.parameters(), att_target.parameters()):
              p_targ.data.mul_(GlobalVars.TAU)
              p_targ.data.add_((1 - GlobalVars.TAU) * p.data)

        