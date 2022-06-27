import numpy as np
from ._globals import GlobalVars

def ff_obs(obs, walls, position, target):
    
    if GlobalVars.IS_ENCODING_PER_AGENT_ENABLED:
    
        WALL_ENCODING = 1;
        EMPTY_ENCODING = 2;
        
        
        POSITION_A1_ENCODING = 3;
        POSITION_A2_ENCODING = 4;
        POSITION_A3_ENCODING = 5;
        POSITION_A4_ENCODING = 6;
        
        TARGET_A1_ENCODING = 7;
        TARGET_A2_ENCODING = 8;
        TARGET_A3_ENCODING = 9;
        TARGET_A4_ENCODING = 10;
        
    else:
        POSITION_ENCODING = 100;
        TARGET_ENCODING = 200;
        WALL_ENCODING = 500;


    if GlobalVars.IS_FF_INPUT_BINARY_VECTORS:
        output = obs
    
    else:
        output = walls * WALL_ENCODING + position * POSITION_ENCODING + target * TARGET_ENCODING
        output[output == (POSITION_ENCODING + TARGET_ENCODING)] = POSITION_ENCODING

    return output



def cnn_obs(walls, position, target):
    
    
    if GlobalVars.IS_ENCODING_PER_AGENT_ENABLED:
    
        WALL_ENCODING = 1;
        EMPTY_ENCODING = 2;
        
        POSITION_A1_ENCODING = 3;
        POSITION_A2_ENCODING = 4;
        POSITION_A3_ENCODING = 5;
        POSITION_A4_ENCODING = 6;
        
        TARGET_A1_ENCODING = 7;
        TARGET_A2_ENCODING = 8;
        TARGET_A3_ENCODING = 9;
        TARGET_A4_ENCODING = 10;
        
    else:
        POSITION_ENCODING = 100;
        TARGET_ENCODING = 200;
        WALL_ENCODING = 500;


    if GlobalVars.IS_CNN_INPUT_BINARY_MATRICES:
        output = np.concatenate((walls, position, target), axis=2)
        
    else:
        if GlobalVars.IS_ENCODING_PER_AGENT_ENABLED:
            
            X = np.array([[POSITION_A1_ENCODING]*100, 
                        [POSITION_A2_ENCODING]*100, 
                        [POSITION_A3_ENCODING]*100,
                        [POSITION_A4_ENCODING]*100], dtype=np.float32)
            
            Y = np.array([[TARGET_A1_ENCODING]*100, 
                        [TARGET_A2_ENCODING]*100, 
                        [TARGET_A3_ENCODING]*100, 
                        [TARGET_A4_ENCODING]*100], dtype=np.float32)
            
                
            output = walls * WALL_ENCODING +\
                position *  X +\
                target *  Y
        else:
            output = walls * WALL_ENCODING + position * POSITION_ENCODING + target * TARGET_ENCODING
            output[output == (POSITION_ENCODING + TARGET_ENCODING)] = POSITION_ENCODING
            
        
    return output
