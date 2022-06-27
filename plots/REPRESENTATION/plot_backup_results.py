"""
backup results
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter


SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

status = 'testing'
backup = '_backup'


model = 'binary_cnn'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    binary_cnn_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    binary_cnn_testing_wall_collisions_over_training_steps = pickle.load(fp)


model = 'numeric_ff'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    numeric_ff_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_ff_testing_wall_collisions_over_training_steps = pickle.load(fp)



model = 'binary_ff'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    binary_ff_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    binary_ff_testing_wall_collisions_over_training_steps = pickle.load(fp)




model = 'numeric_cnn'
with open ('./'+model+'/'+status+'_agent_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_agent_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_collisions_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_coordinations_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_coordinations_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_cumulative_rewards'+backup,'rb') as fp:
    numeric_cnn_testing_cumulative_rewards = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_length_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_episode_length_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_episode_steps_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_episode_steps_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_goals_reached_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_goals_reached_over_training_steps = pickle.load(fp)
with open ('./'+model+'/'+status+'_wall_collisions_over_training_steps'+backup,'rb') as fp:
    numeric_cnn_testing_wall_collisions_over_training_steps = pickle.load(fp)


start = 999
print('rewards')
print('binary ff      ',
      round(np.array(binary_ff_testing_cumulative_rewards)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_cumulative_rewards)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_cumulative_rewards)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_cumulative_rewards)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_cumulative_rewards)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_cumulative_rewards)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_cumulative_rewards)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_cumulative_rewards)[start:1000].std(), 3))


print('length')
print('binary ff new     ',
      round(np.array(binary_ff_testing_episode_steps_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_episode_steps_over_training_steps)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_episode_steps_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_episode_steps_over_training_steps)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_episode_steps_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_episode_steps_over_training_steps)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_episode_steps_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_episode_steps_over_training_steps)[start:1000].std(), 3))


print('pess length')
print('binary ff new     ',
      round(np.array(binary_ff_testing_episode_length_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_episode_length_over_training_steps)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_episode_length_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_episode_length_over_training_steps)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_episode_length_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_episode_length_over_training_steps)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_episode_length_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_episode_length_over_training_steps)[start:1000].std(), 3))


print('reached')
print('binary ff new     ',
      round(np.array(binary_ff_testing_goals_reached_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_goals_reached_over_training_steps)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_goals_reached_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_goals_reached_over_training_steps)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_goals_reached_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_goals_reached_over_training_steps)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_goals_reached_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_goals_reached_over_training_steps)[start:1000].std(), 3))


print('collisions')
print('binary ff new     ',
      round(np.array(binary_ff_testing_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_collisions_over_training_steps)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_collisions_over_training_steps)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_collisions_over_training_steps)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_collisions_over_training_steps)[start:1000].std(), 3))


print('wall coll')
print('binary ff new     ',
      round(np.array(binary_ff_testing_wall_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_wall_collisions_over_training_steps)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_wall_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_wall_collisions_over_training_steps)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_wall_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_wall_collisions_over_training_steps)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_wall_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_wall_collisions_over_training_steps)[start:1000].std(), 3))


print('agent coll')
print('binary ff new     ',
      round(np.array(binary_ff_testing_agent_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_ff_testing_agent_collisions_over_training_steps)[start:1000].std(), 3))
print('numeric cnn       ', 
      round(np.array(numeric_cnn_testing_agent_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_cnn_testing_agent_collisions_over_training_steps)[start:1000].std(), 3))
print('binary cnn       ', 
      round(np.array(binary_cnn_testing_agent_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(binary_cnn_testing_agent_collisions_over_training_steps)[start:1000].std(), 3))
print('ff numeric       ', 
      round(np.array(numeric_ff_testing_agent_collisions_over_training_steps)[start:1000].mean(), 3), 
      round(np.array(numeric_ff_testing_agent_collisions_over_training_steps)[start:1000].std(), 3))



smoothin_window = 11
smoothing_ = 9
LABEL = '-[BACKUP]'


def plotting(id):
    fig, a = plt.subplots(1,id)
    plt.grid(color='k', linestyle='-', linewidth=0.25)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    return fig
        

fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_CumulativeReward.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_cumulative_rewards).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Cumulative Rewards') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_CumulativeReward1.png', dpi=300)



fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_Collisions.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Collisions (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_Collisions1.png', dpi=300)




fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_Coordinations.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_coordinations_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Coordinations (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_Coordinations1.png', dpi=300)




fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_EpisodeLength.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_steps_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (individual)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_EpisodeLength1.png', dpi=300)



fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_GoalsReached.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_goals_reached_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Task Done (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_GoalsReached1.png', dpi=300)



fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_agent_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Miscoordinations (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_MisCoordinations1.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_wall_collisions_over_training_steps).mean(axis=1), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Wall Collisions (%)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_WallCollisions1.png', dpi=300)



fig = plotting(1)
plt.plot(np.array(numeric_ff_testing_episode_length_over_training_steps), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_ff_testing_episode_length_over_training_steps), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(numeric_cnn_testing_episode_length_over_training_steps), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(np.array(binary_cnn_testing_episode_length_over_training_steps), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_EpisodeLengthPess.png', dpi=300)


fig = plotting(1)
plt.plot(savgol_filter(np.array(numeric_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#EBF377', label='BaseATOC-MLP-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_ff_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#E36A7E', label='BaseATOC-MLP-B'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(numeric_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#5B86FA', label='BaseATOC-CNN-N'+LABEL, alpha=1, linewidth=0.8)
plt.plot(savgol_filter(np.array(binary_cnn_testing_episode_length_over_training_steps), smoothin_window, smoothing_), '#76E14C', label='BaseATOC-CNN-B'+LABEL, alpha=1, linewidth=0.8)
plt.xlabel('Steps (×4e3)')
plt.ylabel('Episode Length (group)') 
plt.legend()
plt.savefig('BACKUP/BackupFourVariantsResults_WallCollisions1_EpisodeLengthPess1.png', dpi=300)


