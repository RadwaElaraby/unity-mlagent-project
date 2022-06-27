# -*- coding: utf-8 -*-

from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
from typing import List
import random
import torch.nn as nn
import sys
from collections import defaultdict
import pickle

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

d=defaultdict(list)
for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
    d[k].append(v)

model = 'atoc'

if model == 'dqn':
    from rh_marl_dqn.model import QNetwork
    from rh_marl_dqn.trainer import Trainer, Buffer
    from rh_marl_dqn._globals import GlobalVars
elif model == 'atoc':
    from rh_marl_atoc.trainer import Trainer, Buffer
    from rh_marl_atoc._globals import GlobalVars
    from rh_marl_atoc.model_cnn import ATOC, ATT, ATT2, CNNATT
    from rh_marl_atoc.evaluator2 import Evaluator
    
if __name__ == "__main__": 
        
    try:
        # This is a non-blocking call that only loads the environment.
        env = UnityEnvironment('../game/MyPathFinder.exe', no_graphics=True, worker_id=GlobalVars.WORKER_ID, seed=1, side_channels=[])

        if model == 'dqn':
            print('model: dqn')
            
            qnet = QNetwork(input_size=GlobalVars.INPUT_SIZE, encoding_size=GlobalVars.ENCODING_SIZE, output_size=GlobalVars.OUTPUT_SIZE)
            
            experiences: Buffer = []
            optim = torch.optim.Adam(qnet.parameters(), lr= 0.001)
            cumulative_rewards: List[float] = []
            
            
            for n in range(GlobalVars.NUM_TRAINING_STEPS):
                print('training...')
                new_exp, _ = Trainer.generate_trajectories(env, qnet, buffer_size=GlobalVars.NUM_NEW_EXP, epsilon=0.1)
                
                random.shuffle(experiences)
                if len(experiences) > GlobalVars.BUFFER_SIZE:
                    experiences = experiences[:GlobalVars.BUFFER_SIZE]
                experiences.extend(new_exp)
                
                print('evaluating...')
                Trainer.update_q_net(qnet, optim, experiences, action_size=GlobalVars.OUTPUT_SIZE)
                                
            torch.save(qnet.state_dict(), './models/[GAME_ID]/qnet.pt')
            
            env.close()
    
    
        elif model == 'atoc':
            print('model: atoc')
    
    
            atoc = ATOC(n_agent=GlobalVars.NUM_AGENTS, num_inputs=GlobalVars.INPUT_SIZE, hidden_dim=GlobalVars.ENCODING_SIZE, num_actions=GlobalVars.OUTPUT_SIZE)
            atoc_target = ATOC(n_agent=GlobalVars.NUM_AGENTS, num_inputs=GlobalVars.INPUT_SIZE, hidden_dim=GlobalVars.ENCODING_SIZE, num_actions=GlobalVars.OUTPUT_SIZE)
            
            atoc = atoc.cuda()
            atoc_target = atoc_target.cuda()
            atoc_target.load_state_dict(atoc.state_dict())
    
            # attention unit
            att = ATT(GlobalVars.NUM_AGENTS, din=GlobalVars.INPUT_SIZE, hidden_dim=GlobalVars.ENCODING_SIZE, dout=GlobalVars.ENCODING_SIZE).cuda()
            att_target = ATT(GlobalVars.NUM_AGENTS, din=GlobalVars.INPUT_SIZE, hidden_dim=GlobalVars.ENCODING_SIZE, dout=GlobalVars.ENCODING_SIZE).cuda()
            
            att_target.load_state_dict(att.state_dict())
            

            
            optim_atoc = torch.optim.RMSprop(atoc.parameters(), lr = 0.0005)
            optim_att = torch.optim.RMSprop(att.parameters(), lr = 0.0005)
            #optim_atoc = torch.optim.Adam(atoc.parameters(), lr = 0.0005)
            #optim_att = torch.optim.Adam(att.parameters(), lr = 0.0005)
            
            criterion_att = nn.BCELoss()
    
    
            experiences: Buffer = []
            training_cumulative_rewards: List[np.ndarray] = []
            training_collisions_over_training_steps: List[np.ndarray] = []
            training_wall_collisions_over_training_steps: List[np.ndarray] = []
            training_agent_collisions_over_training_steps: List[np.ndarray] = []
            training_coordinations_over_training_steps: List[np.ndarray] = []
            training_episode_length_over_training_steps: List[float] = []
            training_goals_reached_over_training_steps: List[float] = []
            training_episode_steps_over_training_steps: List[float] = []

            testing_cumulative_rewards: List[np.ndarray] = []
            testing_collisions_over_training_steps: List[np.ndarray] = []
            testing_wall_collisions_over_training_steps: List[np.ndarray] = []
            testing_agent_collisions_over_training_steps: List[np.ndarray] = []
            testing_coordinations_over_training_steps: List[np.ndarray] = []
            testing_episode_length_over_training_steps: List[float] = []
            testing_goals_reached_over_training_steps: List[np.ndarray] = []
            testing_episode_steps_over_training_steps: List[np.ndarray] = []

            testing_cumulative_rewards_backup: List[np.ndarray] = []
            testing_collisions_over_training_steps_backup: List[np.ndarray] = []
            testing_wall_collisions_over_training_steps_backup: List[np.ndarray] = []
            testing_agent_collisions_over_training_steps_backup: List[np.ndarray] = []
            testing_coordinations_over_training_steps_backup: List[np.ndarray] = []
            testing_episode_length_over_training_steps_backup: List[float] = []
            testing_goals_reached_over_training_steps_backup: List[np.ndarray] = []
            testing_episode_steps_over_training_steps_backup: List[np.ndarray] = []
            
            epsilon = 0.9
            min_epsilon = 0.1
            epsilon_delta = (epsilon - min_epsilon) / GlobalVars.NUM_TRAINING_STEPS
    
            for n in range(GlobalVars.NUM_TRAINING_STEPS+750): #GlobalVars.NUM_TRAINING_STEPS+200+300
                
                print('training...')
                
                new_exp, training_rewards, training_collisions, training_wall_collisions, training_agent_collisions, training_coordinations, training_length, training_goals_reached, training_episode_steps = Trainer.generate_trajectories(env, atoc, att, buffer_size=GlobalVars.NUM_NEW_EXP, epsilon=epsilon)

        
                training_cumulative_rewards.append(training_rewards)
                training_collisions_over_training_steps.append(training_collisions)
                training_wall_collisions_over_training_steps.append(training_wall_collisions)
                training_agent_collisions_over_training_steps.append(training_agent_collisions)
                training_coordinations_over_training_steps.append(training_coordinations)
                training_episode_length_over_training_steps.append(training_length)
                training_goals_reached_over_training_steps.append(training_goals_reached)
                training_episode_steps_over_training_steps.append(training_episode_steps)

                print(n+1, "\treward ", training_rewards, "\tcollisions ", training_collisions, "\twall collisions ", training_wall_collisions, "\tagent collisions ", training_agent_collisions, "\tcoordinations ", training_coordinations, "\tlength ", training_length, "\tgoals ", training_goals_reached, "\t steps/a ", training_episode_steps)

                random.shuffle(experiences)
                if len(experiences) > GlobalVars.BUFFER_SIZE:
                    experiences = experiences[:GlobalVars.BUFFER_SIZE]
                experiences.extend(new_exp)
                
                print('evaluating...')
                
                Trainer.update_atoc_net(atoc, atoc_target, att, att_target, optim_atoc, optim_att, criterion_att, experiences, action_size=GlobalVars.OUTPUT_SIZE)

                _, testing_rewards, testing_collisions, testing_wall_collisions, testing_agent_collisions, testing_coordinations, testing_length, testing_goals_reached, testing_episode_steps = Trainer.generate_trajectories(env, atoc, att, buffer_size=1000, epsilon=0)
                    
                testing_cumulative_rewards.append(testing_rewards)
                testing_collisions_over_training_steps.append(testing_collisions)
                testing_wall_collisions_over_training_steps.append(testing_wall_collisions)
                testing_agent_collisions_over_training_steps.append(testing_agent_collisions)
                testing_coordinations_over_training_steps.append(testing_coordinations)
                testing_episode_length_over_training_steps.append(testing_length)
                testing_goals_reached_over_training_steps.append(testing_goals_reached)
                testing_episode_steps_over_training_steps.append(testing_episode_steps)
                
                print(n+1, "naive \treward ", testing_rewards, "\tcollisions ", testing_collisions, "\twall collisions ", testing_wall_collisions, "\tagent collisions ", testing_agent_collisions, "\tcoordinations ", testing_coordinations, "\tlength ", testing_length, "\tgoals ", testing_goals_reached, "\t steps/a ", testing_episode_steps)
                
                
                _, testing_rewards_backup, testing_collisions_backup, testing_wall_collisions_backup, testing_agent_collisions_backup, testing_coordinations_backup, testing_length_backup, testing_goals_reached_backup, testing_episode_steps_backup = Evaluator.generate_trajectories(env, atoc, att, buffer_size=1000, epsilon=0)

                testing_cumulative_rewards_backup.append(testing_rewards_backup)
                testing_collisions_over_training_steps_backup.append(testing_collisions_backup)
                testing_wall_collisions_over_training_steps_backup.append(testing_wall_collisions_backup)
                testing_agent_collisions_over_training_steps_backup.append(testing_agent_collisions_backup)
                testing_coordinations_over_training_steps_backup.append(testing_coordinations_backup)
                testing_episode_length_over_training_steps_backup.append(testing_length_backup)
                testing_goals_reached_over_training_steps_backup.append(testing_goals_reached_backup)
                testing_episode_steps_over_training_steps_backup.append(testing_episode_steps_backup)

                print(n+1, "backup \treward ", testing_rewards_backup, "\tcollisions ", testing_collisions_backup, "\twall collisions ", testing_wall_collisions_backup, "\tagent collisions ", testing_agent_collisions_backup, "\tcoordinations ", testing_coordinations_backup, "\tlength ", testing_length_backup, "\tgoals ", testing_goals_reached_backup, "\t steps/a ", testing_episode_steps_backup)
                
                
                epsilon = max(epsilon - epsilon_delta, 0.1)
       
                print('------------------------------------------------------------------------------------')
                
      
                
            torch.save(atoc.state_dict(), './output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/atoc.pt')
            torch.save(att.state_dict(), './output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/att.pt')
        
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_cumulative_rewards', 'wb') as fp:
                pickle.dump(training_cumulative_rewards, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_collisions_over_training_steps', 'wb') as fp:
                pickle.dump(training_collisions_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_wall_collisions_over_training_steps', 'wb') as fp:
                pickle.dump(training_wall_collisions_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_agent_collisions_over_training_steps', 'wb') as fp:
                pickle.dump(training_agent_collisions_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_coordinations_over_training_steps', 'wb') as fp:
                pickle.dump(training_coordinations_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_episode_length_over_training_steps', 'wb') as fp:
                pickle.dump(training_episode_length_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_goals_reached_over_training_steps', 'wb') as fp:
                pickle.dump(training_goals_reached_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/training_episode_steps_over_training_steps', 'wb') as fp:
                pickle.dump(training_episode_steps_over_training_steps, fp)
                
                
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_cumulative_rewards', 'wb') as fp:
                pickle.dump(testing_cumulative_rewards, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_collisions_over_training_steps', 'wb') as fp:
                pickle.dump(testing_collisions_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_wall_collisions_over_training_steps', 'wb') as fp:
                pickle.dump(testing_wall_collisions_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_agent_collisions_over_training_steps', 'wb') as fp:
                pickle.dump(testing_agent_collisions_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_coordinations_over_training_steps', 'wb') as fp:
                pickle.dump(testing_coordinations_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_length_over_training_steps', 'wb') as fp:
                pickle.dump(testing_episode_length_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_goals_reached_over_training_steps', 'wb') as fp:
                pickle.dump(testing_goals_reached_over_training_steps, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_steps_over_training_steps', 'wb') as fp:
                pickle.dump(testing_episode_steps_over_training_steps, fp)

            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_cumulative_rewards_backup', 'wb') as fp:
                pickle.dump(testing_cumulative_rewards_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_collisions_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_collisions_over_training_steps_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_wall_collisions_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_wall_collisions_over_training_steps_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_agent_collisions_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_agent_collisions_over_training_steps_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_coordinations_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_coordinations_over_training_steps_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_length_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_episode_length_over_training_steps_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_goals_reached_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_goals_reached_over_training_steps_backup, fp)
            with open('output/'+GlobalVars.GAME_NAME+'/'+GlobalVars.COMMUNICATION_TYPE+'/testing_episode_steps_over_training_steps_backup', 'wb') as fp:
                pickle.dump(testing_episode_steps_over_training_steps_backup, fp)

               
                
            env.close()
            
            
    except Exception as e:
        env.close()
        raise
        

 