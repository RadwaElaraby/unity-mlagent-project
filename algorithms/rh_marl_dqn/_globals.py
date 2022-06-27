# -*- coding: utf-8 -*-

class GlobalVars:

    USE_STATIC_POLICY = True # argmax or softmax
    
    WORKER_ID = 5306
    
    ACTION_SIZE = 1 # 1 numeric value range between 0-4 for each agent
    
    NUM_AGENTS = 2
    
    """
    q model settings
    """
    INPUT_SIZE = 36
    
    ENCODING_SIZE = 50
    
    OUTPUT_SIZE = 5
    
    
    """
    Training
    """
    # The number of training steps that will be performed
    NUM_TRAINING_STEPS = 100
    
    # The number of experiences to collect per training step
    NUM_NEW_EXP = 1000 # 1000
    
    # The maximum size of the Buffer
    BUFFER_SIZE = 10000
    


