import numpy as np
from typing import NamedTuple, List

class DecisionStep():
    def __init__(self, obs, reward, agent_id, action_mask, group_id, group_reward, comm_region):
        self.obs = obs
        self.reward = reward
        self.agent_id = agent_id 
        self.action_mask = action_mask
        self.group_id = group_id
        self.group_reward = group_reward
        self.comm_region = comm_region
        
                     
class TerminalStep():
    def __init__(self, obs, reward, interrupted, agent_id, group_id, group_reward, comm_region):
        self.obs = obs
        self.reward = reward
        self.agent_id = agent_id 
        self.interrupted = interrupted
        self.group_id = group_id
        self.group_reward = group_reward
        self.comm_region = comm_region


class Experience(NamedTuple):
  obs: np.ndarray
  action: np.ndarray
  reward: float
  next_obs: np.ndarray
  done: bool
  comm_region: np.ndarray
  next_comm_region: np.ndarray
  action_mask: np.ndarray
  next_action_mask: np.ndarray

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]

# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]


