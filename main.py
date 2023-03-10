import random
import numpy as np
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.signal import savgol_filter
from uav_env import UAVenv
from misc import final_render
import os
import math

os.chdir = ("")

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

class Q_Learning:
    def __init__(self):
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.steps_done = 0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.state_size = 2
        self.action_size = 5
        self.COVERAGE_XY = 1000
        self.grid_space = 100
        self.GRID_SIZE = int(self.COVERAGE_XY / self.grid_space)  # Each grid defined as 100m block
        self.state_space_size = (self.GRID_SIZE, self.GRID_SIZE)
        self.action_space_size = 5
        self.gamma = discount_factor
        # For 10x10 Grid there are 11 possisblity in each direction from 0 to 10
        self.Q = np.zeros((self.state_space_size[0] + 1, self.state_space_size[1] + 1) * NUM_UAV)

    # Deployment of epsilon greedy policy
    def epsilon_greedy(self, state):
        temp = random.random()
        # Epsilon decay policy is employed for faster convergence
        epsilon_thres = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1*self.steps_done/self.epsilon_decay)
        self.steps_done += 1 
        # Compare against a epsilon threshold to either explore or exploit
        if temp <= epsilon_thres:
            # If less than threshold action choosen randomly
            action = np.random.randint(0, 4)
        else:
            # Else (high prob) choosing the best possible action giving maximum Q-value
            # print(self.Q[state[0], state[1], :])
            action = np.argmax(self.Q[state[0], state[1], :])
        return action

    # Computation of Q-values of state-action pair
    def qlearning(self, info):
        # Info should be in the following format
        # Info = (state, action, next_sta, reward_ind )
        state = info[0].astype(int)
        action = info[1]
        next_sta = info[2].astype(int)
        reward = info[3]
        best_next_action = np.argmax(self.Q[next_sta[0], next_sta[1], :])
        td_target = reward + self.gamma * self.Q[next_sta[0], next_sta[1], best_next_action]
        td_delta = td_target - self.Q[state[0], state[1], action]
        self.Q[state[0], state[1], action] += self.alpha * td_delta


u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 800
num_epochs = 100
discount_factor = 0.95
alpha = 0.5
batch_size = 512
update_rate = 10  #50
dnn_epoch = 1
epsilon = 0.1
epsilon_min = 0.1
epsilon_decay = 1
random.seed(SEED)

## The whole problem of no convergence and no oscillation was dues to really low learning rate, as a result it was working 
## really slowly as a result it was taking way to long to converge to more optimal position
## Having a higher learning rate helped in this case 

# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)
episode_user_connected = np.zeros(num_episode)

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])

UAV_OB = [None, None, None, None, None]

for k in range(NUM_UAV):
            UAV_OB[k] = Q_Learning()
best_result = 0

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])

for i_episode in range(num_episode):
    print(i_episode)

    # Environment reset and get the states
    u_env.reset()

    # Get the initial states
    states = u_env.get_state()
    reward = np.zeros(NUM_UAV)

    
    for t in range(num_epochs):
        drone_act_list = []

        # Determining the actions for all drones
        states_ten = states
        for k in range(NUM_UAV):
            state = states_ten[k, :].astype(int)
            action = UAV_OB[k].epsilon_greedy(state)
            drone_act_list.append(action)

        # Find the rewards for the combined set of actions for the UAV
        temp_data = u_env.step(drone_act_list)
        reward = temp_data[1]
        done = temp_data[2]
        next_state = u_env.get_state()

        # Calculation of the total episodic reward of all the UAVs 
        # Calculation of the total number of connected User for the combination of all UAVs
        episode_reward[i_episode] += sum(reward)
        episode_user_connected[i_episode] += temp_data[4]

        for k in range(NUM_UAV):
            state = states_ten[k, :].astype(int)
            action = drone_act_list[k]
            next_sta = next_state[k, :]
            reward_ind = reward[k]
            # Info = (state, action, next_sta, next_act, reward_ind )
            info = [state, action, next_sta, reward_ind]
            UAV_OB[k].qlearning(info)
            

        states = next_state
        
        # If done break from the loop (go to next episode)
        # if done:
        #     break

    if i_episode % 10 == 0:
        # Reset of the environment
        u_env.reset()
        # Get the states
        # Get the states
        states = u_env.get_state()
        for t in range(100):
            drone_act_list = []
            for k in range(NUM_UAV):
                state = states[k,:].astype(int)
                best_next_action = np.argmax(UAV_OB[k].Q[state[0], state[1], :])
                drone_act_list.append(best_next_action)
            temp_data = u_env.step(drone_act_list)
            states = u_env.get_state()
            states_fin = states
            if best_result < temp_data[4]:
                best_result = temp_data[4]
                best_state = states
        u_env.render(ax1)
            # plt.title("Simulation")
            # plt.savefig(r'C:\Users\tripats\Documents\Results_SameParams0017\Simulations\simul' + str(i_episode)  + str(t) + '.png')
            # print(drone_act_list)
            # print("Number of user connected in ",i_episode," episode is: ", temp_data[4])

def smooth(y, pts):
    box = np.ones(pts)/pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

## Save the data from the run as a file
mdict = {'num_episode':range(0, num_episode),'episodic_reward': episode_reward}
savemat(r'C:\Users\tripats\Documents\GitHub\UAV_Subband_Allocation_QLearning\Result\Run002\Position Information with Distance Penalty - Level 3\episodic_reward.mat', mdict)
mdict_2 = {'num_episode':range(0, num_episode),'connected_user': episode_user_connected}
savemat(r'C:\Users\tripats\Documents\GitHub\UAV_Subband_Allocation_QLearning\Result\Run002\Position Information with Distance Penalty - Level 3\connected_user.mat', mdict_2)


# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_reward)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Episode vs Episodic Reward")
plt.show()
fig = plt.figure()
plt.plot(range(0, num_episode), episode_user_connected)
plt.xlabel("Episode")
plt.ylabel("Connected User in Episode")
plt.title("Episode vs Connected User in Epsisode")
plt.show()
fig = plt.figure()
smoothed = smooth(episode_reward, 10)
plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Smoothed Epidode vs Episodic Reward")
plt.show()
fig = plt.figure()
final_render(states_fin, "final")
fig = plt.figure()
final_render(best_state, "best")
print(states_fin)
print('Total Connected User in Final Stage', temp_data[4])
print("Best State")
print(best_state)
print("Total Connected User (Best Outcome)", best_result)



