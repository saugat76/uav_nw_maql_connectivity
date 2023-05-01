from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'Times New Roman',
        'size' : 12}

matplotlib.rc('font', **font)

pt = 5

def smooth(y, pts):
    box = np.ones(pts)/pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_episode_reward(learning_rate):
    file_path = r'C:\Users\tripats\Documents\GitHub\Results_DQN_Pytorch\Dynamic_Environment\Run002_QLearning\Run903\Learning_Rate_' + str(learning_rate) + r'\episodic_reward.mat'
    episode_reward = loadmat(file_path)
    num_episode = list(episode_reward['num_episode'])
    episode_reward = list(episode_reward['episodic_reward'])
    return num_episode[0], smooth(episode_reward[0], pt)

fig = plt.figure()
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i in range(6):
    learning_rate = (i+1)/10
    num_episode, episode_reward = get_episode_reward(learning_rate)
    plt.plot(num_episode[0:830], episode_reward[0:830], colors[i], label='Learning Rate: ' + str(learning_rate), alpha=0.8)

plt.legend(loc="lower right")
plt.xlabel("Episode", fontsize=14, family='Times New Roman')
plt.ylabel("Episodic Reward", fontsize=14, family='Times New Roman')
plt.title("Episodic Reward vs Episode with Implicit Info Exchange: MAQL")
plt.show()