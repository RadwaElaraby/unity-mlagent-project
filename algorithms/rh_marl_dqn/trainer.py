import numpy as np
import torch
from typing import NamedTuple, List
from typing import Dict
import random
from .model import QNetwork
from mlagents_envs.environment import ActionTuple, BaseEnv
from ._globals import GlobalVars
from .unity_integrate import separate_steps


class Experience(NamedTuple):
  obs: np.ndarray
  action: np.ndarray
  next_obs: np.ndarray
  reward: float
  done: bool

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]

# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]


class Trainer:
  @staticmethod
  def generate_trajectories(env: BaseEnv, q_net: QNetwork, buffer_size: int, epsilon: float):
    # Create an empty Buffer
    buffer: Buffer = []
    
    # Reset the environment
    env.reset()
    # Read and store the Behavior Name of the Environment
    behavior_name = list(env.behavior_specs)[0]
    # Read and store the Behavior Specs of the Environment
    spec = env.behavior_specs[behavior_name]

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

    while len(buffer) < buffer_size:  # While not enough data in the buffer
    
      #print('----------------------------------------------------------------')
                  
      decision_steps, terminal_steps = env.get_steps(behavior_name)
      
      decision_steps, terminal_steps = separate_steps(decision_steps, terminal_steps)
      
        # For all Agents with a Terminal Step:
            
      for agent_id_terminated in terminal_steps: #range(len(terminal_steps)):
        if agent_id_terminated not in dict_last_action_from_agent:
            continue
      
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
        buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
        buffer.append(last_experience)

      
      # For all Agents with a Decision Step:
      for agent_id_decisions in decision_steps: #range(len(decision_steps)):
        # If the Agent does not have a Trajectory, create an empty one
        if agent_id_decisions not in dict_trajectories_from_agent:
          dict_trajectories_from_agent[agent_id_decisions] = []
          dict_cumulative_reward_from_agent[agent_id_decisions] = 0


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
          # Update the Trajectory of the Agent and its cumulative reward
          dict_trajectories_from_agent[agent_id_decisions].append(exp)
          dict_cumulative_reward_from_agent[agent_id_decisions] += (
            decision_steps[agent_id_decisions].group_reward + decision_steps[agent_id_decisions].reward
          )
          
        # Store the observation as the new "last observation"
        dict_last_obs_from_agent[agent_id_decisions] = (
          decision_steps[agent_id_decisions].obs[0]
        )
        
      #print('decision_steps.obs[0]')
      #print(decision_steps.obs[0])
      
      # Generate an action for all the Agents that requested a decision
      # Compute the values for each action given the observation

      #all_obs = np.array(list(map(lambda d: d.obs[0:INPUT_SIZE], decision_steps)))
      all_obs = np.array([v.obs[0][0:GlobalVars.INPUT_SIZE] for k,v in decision_steps.items()])
      
      
      # if some agents have already terminated, add zeros for obs
      # they won't be used. it's just there to maintain shape of actions
      if (len(all_obs) < GlobalVars.NUM_AGENTS):
          adjusted_obs = np.zeros((GlobalVars.NUM_AGENTS, GlobalVars.INPUT_SIZE)).astype(np.float32)
          for agent_id in range(GlobalVars.NUM_AGENTS):
              if agent_id in decision_steps:
                  adjusted_obs[agent_id,] = all_obs[0,:]
                  np.delete(all_obs, (0), axis=0) # remove first row
          all_obs = adjusted_obs   

      actions_values = (
      #q_net(torch.from_numpy(decision_steps.obs[0])).detach().numpy()
          q_net(torch.from_numpy(all_obs)).detach().numpy()
      )
      
      if random.random() <= epsilon:
          # Add some noise with epsilon to the values
          actions_values = (
              np.random.randn(actions_values.shape[0], actions_values.shape[1])
          ).astype(np.float32)
      
      # Pick the best action using argmax
      if GlobalVars.USE_STATIC_POLICY:
          # static policy 
          actions = np.argmax(actions_values, axis=1)
      else:
          # random policy
          action_list = np.array(list(range(GlobalVars.OUTPUT_SIZE))).astype(np.int64)
          actions = np.array([random.choices(action_list, weights=x) for x in actions_values])
      
      
      actions.resize((GlobalVars.NUM_AGENTS, 1))

      # Store the action that was picked, it will be put in the trajectory later
      #for agent_index, agent_id in enumerate(decision_steps.agent_id):
      all_agent_ids = np.array([v.agent_id for k,v in decision_steps.items()])
      for agent_id in all_agent_ids:
        # the stored action should be an array and not just a value
        dict_last_action_from_agent[agent_id] = actions[agent_id]


      # Set the actions in the environment
      # Unity Environments expect ActionTuple instances.
      actions.resize((1, GlobalVars.NUM_AGENTS * GlobalVars.ACTION_SIZE)) # 1 because centralized
      action_tuple = ActionTuple()
      action_tuple.add_discrete(actions)
      env.set_actions(behavior_name, action_tuple)

      
      # Perform a step in the simulation
      env.step()
    return buffer, np.mean(cumulative_rewards)

  @staticmethod
  def update_q_net(q_net: QNetwork, optimizer: torch.optim, buffer: Buffer, action_size: int):
    BATCH_SIZE = 1000
    NUM_EPOCH = 3
    GAMMA = 0.9
    batch_size = min(len(buffer), BATCH_SIZE)
    
    random.shuffle(buffer)
    
    # Split the buffer into batches
    batches = [
      buffer[batch_size * start : batch_size * (start + 1)]
      for start in range(int(len(buffer) / batch_size))
    ]
    
    for _ in range(NUM_EPOCH):
      for batch in batches:
        # Create the Tensors that will be fed in the network
        obs = torch.from_numpy(
            np.stack([ex.obs for ex in batch])
        )
        action = torch.from_numpy(
            np.stack([ex.action for ex in batch])
        )
        next_obs = torch.from_numpy(
            np.stack([ex.next_obs for ex in batch])
        )
        reward = torch.from_numpy(
          np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
        )
        done = torch.from_numpy(
          np.array([ex.done for ex in batch], dtype=np.float32).reshape(-1, 1)
        )

        # Use the Bellman equation to update the Q-Network
        target = (
          reward
          + (1.0 - done)
          * GAMMA
          * torch.max(q_net(next_obs).detach(), dim=1, keepdim=True).values
        )
        
        mask = torch.zeros((len(batch), action_size))
        mask.scatter_(1, action, 1)
        prediction = torch.sum(q_net(obs) * mask, dim=1, keepdim=True)
        
        criterion = torch.nn.MSELoss()
        loss = criterion(prediction, target)

        # Perform the backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
