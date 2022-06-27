# -*- coding: utf-8 -*-

import numpy as np
from ._globals import GlobalVars

import random
def am_i_colliding_with_wall(current_walls, current_pos, action):
    
    row, col = np.where(current_pos == 1)
    current_row = row[0]
    current_col = col[0]
    
    
    if action == 0:
        new_row = current_row
        new_col = current_col
        pass
    
    elif action == 1: # up
        if current_row-1 < 0:
            raise
        new_row = current_row-1
        new_col = current_col
        
    elif action == 2: # down
        if current_row+1 >= GlobalVars.GRID_HEIGHT:
            raise
        new_row = current_row+1
        new_col = current_col
      
    elif action == 3: # right
        if current_col+1 >= GlobalVars.GRID_WIDTH:
            raise
        new_row = current_row
        new_col = current_col+1
        
    elif action == 4: # left
        if current_col-1 < 0:
            raise
        new_row = current_row
        new_col = current_col-1

    if current_walls[new_row, new_col] == 0:
        return False
    
    return True
    
def calculate_current_next_positions(all_obs, actions, agents_terminated, agent_id_decisions):
    next_positions = []
    current_positions = []
    for i in range(GlobalVars.NUM_AGENTS):
        curr_pos = all_obs[agent_id_decisions][i][GlobalVars.CURRENT_POSITION_MAP_INDEX : GlobalVars.CURRENT_POSITION_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
        
        if agents_terminated[i]:
            curr_pos = curr_pos.flatten()
            curr_pos = np.where(curr_pos == 1)[0][0]
            next_positions.append(curr_pos)
            current_positions.append(curr_pos)
        else:
            curr_walls = all_obs[agent_id_decisions][i][GlobalVars.WALL_MAP_INDEX : GlobalVars.WALL_MAP_INDEX+GlobalVars.GRID_SIZE].reshape(GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
            curr_action = actions[i]
            new_pos = action_to_new_pos(curr_walls, curr_pos, curr_action)
            new_pos = new_pos.flatten()
            next_positions.append(np.where(new_pos == 1)[0][0])
            curr_pos = curr_pos.flatten()
            current_positions.append(np.where(curr_pos == 1)[0][0])
            
    return current_positions, next_positions
def am_i_colliding_with_another_agent(idx, current_positions, next_positions):
         
    who_else = [i for i, x in enumerate(next_positions) if x == next_positions[idx]]                    
    if len(who_else) > 1:
        if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
            print(who_else, 'trying to go to the same position!')
        return True
    
    who_switch = [i for i, (f,t) in enumerate(zip(current_positions, next_positions)) 
                  if current_positions[idx] != next_positions[idx] 
                  and t == current_positions[idx] and f == next_positions[idx]]
    if len(who_switch) > 0:
        if GlobalVars.SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES:
            print(who_switch, 'trying to switch!')
        return True
    return False




def action_to_new_pos(current_walls, current_pos, action):
    # from 0 to 4
    # 0 no action
    # 1 up 
    # 2 down
    # 3 right
    # 4 left

    new_pos = np.copy(current_pos)
    
    row, col = np.where(current_pos == 1)
    current_row = row[0]
    current_col = col[0]
    
    
    if action == 0:
        new_row = current_row
        new_col = current_col
        pass
    
    elif action == 1: # up
        if current_row-1 < 0:
            raise
        new_row = current_row-1
        new_col = current_col
        
    elif action == 2: # down
        if current_row+1 >= GlobalVars.GRID_HEIGHT:
            raise
        new_row = current_row+1
        new_col = current_col
      
    elif action == 3: # right
        if current_col+1 >= GlobalVars.GRID_WIDTH:
            raise
        new_row = current_row
        new_col = current_col+1
        
    elif action == 4: # left
        if current_col-1 < 0:
            raise
        new_row = current_row
        new_col = current_col-1
       
    if current_walls[new_row, new_col] == 0:
        new_pos[current_row, current_col] = 0
        new_pos[new_row, new_col] = 1
        
        
    return new_pos


