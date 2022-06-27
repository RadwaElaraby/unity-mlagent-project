# -*- coding: utf-8 -*-

class GlobalVars:

    WORKER_ID = 3769
    
    #USE_STATIC_POLICY = False # argmax or softmax


    ENCODING_SIZE = 200 # 100
    
    # The number of experiences to collect per training step
    NUM_NEW_EXP = 4000 # 1000 
    
    #Game0_7x7_Static
    """
    GAME_NAME = 'Game0_7x7_Static'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 7
    GRID_HEIGHT = 7
    NUM_AGENTS = 2    
    MAX_STEPS_PER_TRAININIG_STEP = 500 
    GOAL_REWARD = 1
    USE_STATIC_POLICY = True
    """
    
    #Game1_9x9_Static
    """
    GAME_NAME = 'Game1_9x9_Static'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 9
    GRID_HEIGHT = 9
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 500 
    GOAL_REWARD = 1
    USE_STATIC_POLICY = True
    """
    
    
    
    
    
    
    
    
    #Game0_7x7_Dynamic
    """
    GAME_NAME = 'Game0_7x7_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 7
    GRID_HEIGHT = 7
    NUM_AGENTS = 2    
    MAX_STEPS_PER_TRAININIG_STEP = 100 
    GOAL_REWARD = 1
    USE_STATIC_POLICY = True
    """
      
    #Game2_10x10_Dynamic
    """
    GAME_NAME = 'Game2_10x10_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 100  
    GOAL_REWARD = 1 # 1 ## althought fixed-length episode, 1 was necessary to simulate agents to move
    USE_STATIC_POLICY = True
    """
    
    #Game1_9x9_Dynamic
    """
    GAME_NAME = 'Game1_9x9_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 9
    GRID_HEIGHT = 9
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 100 
    GOAL_REWARD = 1
    USE_STATIC_POLICY = True
    """
       
    

    
    
    
    
    #Game2_10x10_Dynamic_Walls
    """

    GAME_NAME = 'Game2_10x10_Dynamic_Walls'    
    COMMUNICATION_TYPE = 'comm_atoc_eval_stuck_collision_v2' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 100 
    GOAL_REWARD = 1 ## althought fixed-length episode, 1 was necessary to simulate agents to move
    USE_STATIC_POLICY = True
    INPUT_SIZE = ((GRID_WIDTH*GRID_HEIGHT)*3) + NUM_AGENTS
    WALL_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*0
    CURRENT_POSITION_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*1
    CURRENT_TARGET_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*2    
    ACTION_IN_STATE_INDEX = None
    IS_ACTION_SHARED = False
    IS_NEXT_OBS_SHARED = False   
       """


    #Game2_10x10_Dynamic_Walls
    GAME_NAME = 'Game2_10x10_Dynamic_Walls'    
    COMMUNICATION_TYPE = 'binary_ff' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = True
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 100 
    GOAL_REWARD = 1 ## althought fixed-length episode, 1 was necessary to simulate agents to move
    USE_STATIC_POLICY = True
    INPUT_SIZE = ((GRID_WIDTH*GRID_HEIGHT)*3) + NUM_AGENTS
    WALL_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*0
    CURRENT_POSITION_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*1
    CURRENT_TARGET_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*2    
    CURRENT_ID_MAP_INDEX = (GRID_WIDTH*GRID_HEIGHT)*3  
    ACTION_IN_STATE_INDEX = None
    IS_ACTION_SHARED = False
    IS_NEXT_OBS_SHARED = False   
    IS_ENCODING_PER_AGENT_ENABLED = False

    IS_INPUT_PREPROCESSING_ENABLED = True
    IS_CNN_ENCODER_ENABLED = False
    IS_CNN_INPUT_BINARY_MATRICES = False # used for CNN with 3 binary matrices
    
    IS_FF_INPUT_BINARY_VECTORS = True # used for FF with 1 numeric vector
    
    
    
    # Without collision
    PENALTY_PER_STEP = -1/MAX_STEPS_PER_TRAININIG_STEP # -1 #  -1/MAX_STEPS_PER_TRAININIG_STEP if FINITE_EPISODE else 0; # -0.0006 -0.0006666667    
    AGENT_COLLISION_PENALTY_PER_STEP = -3/MAX_STEPS_PER_TRAININIG_STEP # -1 #  -1/MAX_STEPS_PER_TRAININIG_STEP if FINITE_EPISODE else 0; # -0.0006 -0.0006666667    
    WALL_COLLISION_PENALTY_PER_STEP = -2/MAX_STEPS_PER_TRAININIG_STEP # -1 #  -1/MAX_STEPS_PER_TRAININIG_STEP if FINITE_EPISODE else 0; # -0.0006 -0.0006666667     
    
    GRID_SIZE = GRID_WIDTH*GRID_HEIGHT


    IS_STUCK_HANDLING_ENABLED = True
    IS_COLLISION_HANDLING_ENABLED = True
    DEQUE_MAX_LENGTH = 11
    STUCK_SAME_THRESHOLD_INDEX = DEQUE_MAX_LENGTH-3 # 8
    FOR_BACK_THRESHOLD_INDEX = DEQUE_MAX_LENGTH-4       # 7
    FOR_BACK_THRESHOLD_INDEX2 = DEQUE_MAX_LENGTH-8 # 3
    COLLISION_THRESHOLD_INDEX = DEQUE_MAX_LENGTH

    BACKUP_CONTROLLER_MAX_CONSEQUENT_STEPS = 10

    


    
    
    #Game0_7x7_Multi_Dynamic
    """
    GAME_NAME = 'Game0_7x7_Multi_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = False
    GRID_WIDTH = 7
    GRID_HEIGHT = 7
    NUM_AGENTS = 2    
    MAX_STEPS_PER_TRAININIG_STEP = 1500 
    GOAL_REWARD = 0.1
    USE_STATIC_POLICY = True
    """
        
    #Game2_10x10_Multi_Dynamic
    """
    GAME_NAME = 'Game2_10x10_Multi_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = False
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 1500 
    GOAL_REWARD = 1 ## althought fixed-length episode, 1 was necessary to simulate agents to move
    USE_STATIC_POLICY = True
    """
    

    #Game2_20x20_Multi_Dynamic
    """
    GAME_NAME = 'Game2_20x20_Multi_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = False
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    NUM_AGENTS = 8    
    MAX_STEPS_PER_TRAININIG_STEP = 5000 
    GOAL_REWARD = 1
    USE_STATIC_POLICY = True      
    """


    """
    ####################################
    ################### not trained yet
    ####################################
    #Game1_9x9_Multi_Dynamic
    GAME_NAME = 'Game1_9x9_Multi_Dynamic'    
    COMMUNICATION_TYPE = 'comm_atoc' # comm_no or comm_atoc or comm_fully    
    FINITE_EPISODE = False
    GRID_WIDTH = 9
    GRID_HEIGHT = 9
    NUM_AGENTS = 4    
    MAX_STEPS_PER_TRAININIG_STEP = 1500 
    GOAL_REWARD = 0.1 # <<<<<<<<<<<<<<<<< haven't run yet
    """




    ALLOW_COMMUNICATION = True
    
    COMMUNICATION_THRESHOLD = 0.1

    ACTION_SIZE = 1 # 1 numeric value range between 0-4 for each agent    

    COMM_REGION_SIZE = NUM_AGENTS*NUM_AGENTS

    
    # actual input size 
    #((9*9)*3)+NUM_AGENTS  (ID)
    # (without reward/finish_flag/comm_region)
    INPUT_SIZE = ((GRID_WIDTH*GRID_HEIGHT)*3) + NUM_AGENTS
                                    
    
    OUTPUT_SIZE = 5
  
    
    """
    Training
    """
    # The number of training steps that will be performed
    NUM_TRAINING_STEPS = 250 # 100
    
    NUM_EPOCH = 5
    
    
    # The maximum size of the Buffer
    BUFFER_SIZE = 10000
    
    
    
    GAMMA = 0.9
    TAU = 0.98
    
    
    SHOW_TEST_SCIPRT_DEBUG_MESSAGES = False
    SHOW_EVALUATOR_SCIPRT_DEBUG_MESSAGES = False
    SHOW_MODEL_DEBUG_MESSAGES = False
    SHOW_MODEL_DEBUG_MESSAGES_ = False


    