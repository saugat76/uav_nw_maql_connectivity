import random
import gym
import numpy as np
import pandas as pd
import sys
import itertools
from collections import defaultdict

from uav_env import UAVenv

u_env = UAVenv()
Q = defaultdict(float)
total_return = defaultdict(float)
N = defaultdict(int)

# first do the reset
state_full = u_env.reset()


# then step function


# def each_uav(ind_state, ind_reward, ind_done, ind_info, i_state, UAV_idx):
# User of QL for the Optimization of Global Reward
# Defining Epsilon Greedy
def epsilon_greedy_policy(state, Q, UAV_idx):
    epsilon = 0.5
    if random.uniform(0, 1) < epsilon:
        return random.randint(1, 5)
    else:
        return max(list(range(1, 5)), key=lambda x: Q[UAV_idx, state, x])

    # Generate an Episode


num_timesteps = 100


def generate_epsiode(Q):
    episode = np.array([])
    num_uav = u_env.NUM_UAV
    full_state = u_env.reset()
    for t in range(num_timesteps):
        action = []
        for i in range(u_env.NUM_UAV):
            state = full_state[i * 3:i * 3 + 2]
            action[i] = epsilon_greedy_policy(state, Q, i)
        next_state, reward, done, info = u_env.step(action)
        for i in range(u_env.NUM_UAV):
            state = full_state[i * 3:i * 3 + 2]
            np.insert(Q([i, (state, action[i], reward)]))
        if done:
            break
        full_state = next_state
    return episode


# Computation of Optimal Policy
num_iterations = 10000
for i in range(num_iterations):
    episode = generate_epsiode(Q)
    for j in range(u_env.NUM_UAV):
        all_state_action_pair = [(s, a) for (s, a, r) in episode[j, :]]
        rewards = [r for (s, a, r) in episode]
        for t, (state, action, _) in enumerate(episode[j, :]):
            if not state in all_state_action_pair[0:t]:
                R = sum(rewards[t, :])
                total_return[(state, action)] += R
                N[(j, state, action)] += 1
                Q[(j, state, action)] = total_return[(state, action)] / N[(j, state, action)]
