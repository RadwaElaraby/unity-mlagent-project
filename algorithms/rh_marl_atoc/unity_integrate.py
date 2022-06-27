import numpy as np
from ._globals import GlobalVars
from .utils import DecisionStep, TerminalStep


already_terminated = []


def separate_steps(decision_steps, terminal_steps):
    new_decision_stepes = dict()
    new_terminal_steps = dict()

    obs = decision_steps[0].obs[0][:-GlobalVars.COMM_REGION_SIZE]

    comm_region = decision_steps[0].obs[0][-GlobalVars.COMM_REGION_SIZE:]
    comm_region.resize((GlobalVars.NUM_AGENTS, GlobalVars.NUM_AGENTS))

    if (len(decision_steps) > 0):
        new_decision_obs = np.array(np.split(obs, GlobalVars.NUM_AGENTS))
        if np.sum(new_decision_obs[:,-2]/GlobalVars.GOAL_REWARD + new_decision_obs[:,-1]) == GlobalVars.NUM_AGENTS and GlobalVars.FINITE_EPISODE:
            new_terminal_steps[0] = TerminalStep(obs=new_decision_obs[:,:GlobalVars.INPUT_SIZE],
                                                     comm_region=comm_region,
                                                     reward=new_decision_obs[:,-2],
                                                     interrupted=False,
                                                     agent_id=list(range(GlobalVars.NUM_AGENTS)),
                                                     group_id=decision_steps[0].group_id,
                                                     group_reward=decision_steps[0].group_reward)            
        else:
            new_decision_obs = np.array(np.split(obs, GlobalVars.NUM_AGENTS))
            new_decision_stepes[0] = DecisionStep(obs=new_decision_obs[:,:GlobalVars.INPUT_SIZE],
                                                 comm_region=comm_region,
                                                 reward=new_decision_obs[:,-2],
                                                 agent_id=list(range(GlobalVars.NUM_AGENTS)),
                                                 action_mask=decision_steps[0].action_mask,
                                                 group_id=decision_steps[0].group_id,
                                                 group_reward=decision_steps[0].group_reward)
    
    
    if (len(terminal_steps) > 0):
        if terminal_steps[0].interrupted == True:
            new_terminal_obs = np.array(np.split(obs, GlobalVars.NUM_AGENTS))
            new_terminal_steps[0] = TerminalStep(obs=new_terminal_obs[:,:GlobalVars.INPUT_SIZE],
                                                     comm_region=comm_region,
                                                     reward=new_terminal_obs[:,-2],
                                                     interrupted=True,
                                                     agent_id=list(range(GlobalVars.NUM_AGENTS)),
                                                     group_id=terminal_steps[0].group_id,
                                                     group_reward=terminal_steps[0].group_reward)    
            
    
    return new_decision_stepes, new_terminal_steps
    



"""
separate_decision_steps

integrate from custom 

@decision_steps

[[[[NOT USED ANYMORE]]]]
"""
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


"""
separate_terminal_steps

@terminal_steps

[[[[NOT USED ANYMORE]]]]
"""
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

