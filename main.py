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


#
# # then step function
# for i in range(10000):
#     print(u_env.step(
#         [random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)]))
#
#
# # def each_uav(ind_state, ind_reward, ind_done, ind_info, i_state, UAV_idx):
# # User of QL for the Optimization of Global Reward, calculation of return based on the global reward
# # Defining Epsilon Greedy
#
#
# def epsilon_greedy_policy(state, Q, UAV_idx):
#     epsilon = 0.5
#     if random.uniform(0, 1) < epsilon:
#         return random.randint(1, 5)
#     else:
#         q_local = Q[UAV_idx]
#         act = max(q_local(state, act, x))
#         return max(list(range(1, 5)), key=lambda x: q_local)
#
#
# num_timesteps = 1000
#
#
# # Generate an Episode
# def generate_epsiode(Q):
#     episode_local = np.array([])
#     num_uav = u_env.NUM_UAV
#     full_state = u_env.reset()
#     for t_in in range(num_timesteps):
#         action_local = np.zeros(shape=(u_env.NUM_UAV, 1))
#         for e in range(u_env.NUM_UAV):
#             state_local = full_state[e * 3:e * 3 + 2]
#             action_local[e] = epsilon_greedy_policy(state_local, Q, e)
#         next_state, reward, done, info = u_env.step(action_local)
#         for e in range(u_env.NUM_UAV):
#             state_local = full_state[e * 3:e * 3 + 2]
#             Q[e].append((state_local, action_local[e], reward))
#         if done:
#             break
#         full_state = next_state
#     return episode_local
#
#
# # Computation of Optimal Policy
# num_iterations = 10000
# for i in range(num_iterations):
#     episode = generate_epsiode(Q)
#     for j in range(u_env.NUM_UAV):
#         all_state_action_pair = [(s, a) for (s, a, r) in episode[j, :]]
#         rewards = [r for (s, a, r) in episode]
#         for t, (state, action, _) in enumerate(episode[j, :]):
#             if not state in all_state_action_pair[0:t]:
#                 R = sum(rewards[t, :])
#                 total_return[(state, action)] += R
#                 N[(j, state, action)] += 1
#                 Q[(j, state, action)] = total_return[(state, action)] / N[(j, state, action)]


def Q_Learning(env, num_episode, num_epoch, discount_factor=0.9, alpha=0.5, epsilon=0.1):
    Q = np.random.rand(NUM_UAV, int(GRID_SIZE * GRID_SIZE), 5)

    # Keeping track of the episode reward
    episode_reward = np.zeros(num_episode)

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, 0:1])
    ax2 = fig.add_subplot(gs[0:1, 1:2])

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
                    action = random.randint(1, 5)
                else:
                    action = np.argmax(Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])])
                    action = action + 1
                drone_act_list.append(action)

            # Find the global reward for the combined set of actions for the UAV
            temp_data = u_env.step(drone_act_list)
            reward = temp_data[1]
            next_state = u_env.get_state()

            # Update of the episodic reward
            episode_reward[i_episode] += reward

            # Use of Temporal Difference Update
            for k in range(NUM_UAV):
                best_next_action = np.argmax(Q[k][int(next_state[k, 0] * GRID_SIZE + next_state[k, 1])])
                td_target = reward + discount_factor * Q[k][int(next_state[k, 0] * GRID_SIZE + next_state[k, 1])][
                    best_next_action]
                td_delta = td_target - Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])][drone_act_list[k]-1]
                Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])][drone_act_list[k]-1] += alpha * td_delta

            states = next_state

            # if i_episode % 10 == 0:
            #     # Reset of the environment
            #     u_env.reset()
            #     # Get the states
            #     states = u_env.get_state()
            #     for t in range(100):
            #         drone_act_list = []
            #         for k in range(NUM_UAV):
            #             action = np.argmax(Q[k][int(states[k, 0] * GRID_SIZE + states[k, 1])])
            #             drone_act_list.append(action)
            #         temp_data = u_env.step(drone_act_list)
            #         states = u_env.get_state()
            #
            #     ax1.imshow(u_env.get_full_obs())
            #     plt.pause(0.5)
            #     plt.draw()
    return Q, episode_reward


# Defining System Parameters
u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 1000
num_epochs = 500
discount_factor = 0.9
alpha = 0.5
epsilon = 0.1

random.seed(10)

Q, episode_rewards = Q_Learning(u_env, num_episode, num_epochs, discount_factor, alpha, epsilon)

mdict = {'Q': Q}
savemat('Q.mat', mdict)

# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_rewards)
plt.show()
