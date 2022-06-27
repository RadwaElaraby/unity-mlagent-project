import numpy as np
from ._globals import GlobalVars


class DecisionStep():

    def __init__(self, obs, reward, agent_id, action_mask, group_id, group_reward):
        self.obs = obs
        self.reward = reward
        self.agent_id = agent_id 
        self.action_mask = action_mask
        self.group_id = group_id
        self.group_reward = group_reward
        
                     
class TerminalStep():

    def __init__(self, obs, reward, interrupted, agent_id, group_id, group_reward):
        self.obs = obs
        self.reward = reward
        self.agent_id = agent_id 
        self.interrupted = interrupted
        self.group_id = group_id
        self.group_reward = group_reward


already_terminated = []


def separate_steps(decision_steps, terminal_steps):
    new_decision_stepes = dict()
    new_terminal_steps = dict()

    if (len(decision_steps) > 0):
        new_decision_obs = np.split(decision_steps[0].obs[0], GlobalVars.NUM_AGENTS)
        
    if (len(terminal_steps) > 0):    
        new_terminal_obs = np.split(terminal_steps[0].obs[0], GlobalVars.NUM_AGENTS)
    
    global already_terminated
    
    if (len(already_terminated) == GlobalVars.NUM_AGENTS):
        already_terminated = []
    
    if len(decision_steps) > 0: 
        for i in range(GlobalVars.NUM_AGENTS):
            
            if i in already_terminated:
                continue
            
            if (new_decision_obs[i][-2] == 1):
                new_terminal_steps[i] = TerminalStep(obs=[new_decision_obs[i][: GlobalVars.INPUT_SIZE]],
                            reward=new_decision_obs[i][-2],
                            interrupted=False,
                            agent_id=i,
                            group_id=decision_steps[0].group_id,
                            group_reward=decision_steps[0].group_reward)   
                already_terminated.append(i)
                continue
            
            new_decision_stepes[i] = DecisionStep(obs=[new_decision_obs[i][: GlobalVars.INPUT_SIZE]],
                     reward=new_decision_obs[i][-2],
                     agent_id=i,
                     action_mask=decision_steps[0].action_mask[i],
                     group_id=decision_steps[0].group_id,
                     group_reward=decision_steps[0].group_reward)    
        
        
    
    if (len(terminal_steps) > 0):    
        for i in range(GlobalVars.NUM_AGENTS):
            if terminal_steps[0].interrupted == True:
                    new_terminal_steps[i] = TerminalStep(obs=[new_terminal_obs[i][: GlobalVars.INPUT_SIZE]],
                         reward=new_terminal_obs[i][-2],
                         interrupted=terminal_steps[0].interrupted,
                         agent_id=i,
                         group_id=terminal_steps[0].group_id,
                         group_reward=terminal_steps[0].group_reward)    
            
            
            
    return new_decision_stepes, new_terminal_steps
    




def separate_decision_steps(decision_steps):
    separated_decision_steps = []
    separated_obs = np.split(decision_steps[0].obs[0], GlobalVars.NUM_AGENTS)

    for i in range(GlobalVars.NUM_AGENTS):

        decision_step = DecisionStep(obs=[separated_obs[i][: GlobalVars.INPUT_SIZE]],
                     reward=separated_obs[i][-2],
                     agent_id=i,
                     action_mask=decision_steps[0].action_mask[i],
                     group_id=decision_steps[0].group_id,
                     group_reward=decision_steps[0].group_reward)    
        
        separated_decision_steps.append(decision_step)
        
    return separated_decision_steps


def separate_terminal_steps(terminal_steps):

    separated_terminal_steps = []
    separated_obs = np.split(terminal_steps[0].obs[0], GlobalVars.NUM_AGENTS)

    for i in range(GlobalVars.NUM_AGENTS):
        terminal_step = TerminalStep(obs=[separated_obs[i][: GlobalVars.INPUT_SIZE]],
                     reward=separated_obs[i][-2],
                     interrupted=terminal_steps[0].interrupted,
                     agent_id=i,
                     group_id=terminal_steps[0].group_id,
                     group_reward=terminal_steps[0].group_reward)    
        
        separated_terminal_steps.append(terminal_step)
        
    return separated_terminal_steps

