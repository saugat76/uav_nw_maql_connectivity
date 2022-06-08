import random
import gym
import numpy as np
import pandas as pd
import sys
import itertools
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.io import savemat

from uav_env import UAVenv

def Q_Learning(env, num_episode, num_epoch, discount_factor, alpha, epsilon):
    Q = np.random.rand(NUM_UAV, int((GRID_SIZE + 1) * (GRID_SIZE + 1)), 5)

    # Keeping track of the episode reward
    episode_reward = np.zeros(num_episode)

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, 0:1])
    # ax2 = fig.add_subplot(gs[0:1, 1:2])

    for i_episode in range(num_episode):
        print(i_episode)
        # Environment reset and get the states
        u_env.reset()
        # Get the initial states
        states = u_env.get_state()
        for t in range(num_epoch):
            drone_act_list = []
            # Determining the actions for all drones
            for k in range(NUM_UAV):
                temp = random.random()
                if temp <= epsilon:
                    action = random.randint(0, 4)
                    action = action + 1
                else:
                    action = np.argmax(Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])])
                    action = action + 1
                drone_act_list.append(action)

            # Find the global reward for the combined set of actions for the UAV
            temp_data = u_env.step(drone_act_list)
            reward = temp_data[1]
            done = temp_data[2]
            next_state = u_env.get_state()
            
            # If done break from the loop (go to next episode)
            # if done:
            #     break

            # Update of the episodic reward
            episode_reward[i_episode] += reward
            # print(reward)

            # Use of Temporal Difference Update
            for k in range(NUM_UAV):
                best_next_action = np.argmax(Q[k][int(next_state[k, 0] * GRID_SIZE + next_state[k, 1])])
                td_target = reward + discount_factor * Q[k][int(next_state[k, 0] * GRID_SIZE + next_state[k, 1])][
                    best_next_action]
                td_delta = td_target - Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])][drone_act_list[k] - 1]
                Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])][drone_act_list[k] - 1] += alpha * td_delta

            states = next_state

        if i_episode % 10 == 0:
            # Reset of the environment
            u_env.reset()
            # Get the states
            states = u_env.get_state()
            for t in range(100):
                drone_act_list = []
                for k in range(NUM_UAV):
                    best_next_action = np.argmax(Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])]) + 1
                    drone_act_list.append(best_next_action)
                temp_data = u_env.step(drone_act_list)
                states = u_env.get_state()
            # Plot the images
            # ax1.imshow(u_env.get_full_obs())
            # # ax2.imshow(u_env.get_joint_obs())
            # plt.pause(0.5)
            # plt.draw()
            u_env.render(ax1)
            

    return Q, episode_reward, states


# Defining System Parameters
u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 100
num_epochs = 500
discount_factor = 0.99
alpha = 0.5
epsilon = 0.1

random.seed(10)

Q, episode_rewards, state = Q_Learning(u_env, num_episode, num_epochs, discount_factor, alpha, epsilon)

mdict = {'Q': Q}
savemat('Q.mat', mdict)

print(state)
# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_rewards)
plt.show()
