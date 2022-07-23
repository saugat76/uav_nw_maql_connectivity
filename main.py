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
import time 
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import wandb
import argparse

def Q_Learning(env, num_episode, num_epoch, discount_factor, alpha, epsilon, min_epsilon):
    Q = np.random.rand(NUM_UAV, int((GRID_SIZE + 1) * (GRID_SIZE + 1)), 5)
    decay_constant = 0.99

    # Keeping track of the episode reward
    episode_reward = np.zeros(num_episode)
    best_result = 0

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, 0:1])

    UAV_OB = []

    for k in range(NUM_UAV):
                UAV_OB.append(Q_Learning())
    best_result = 0

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
            
            # Find the global reward for the combined set of actions for the UAV
            temp_data = u_env.step(drone_act_list)
            reward = temp_data[1]
            done = temp_data[2]
            next_state = u_env.get_state()

            for k in range(NUM_UAV):
                state = states_ten[k, :].astype(int)
                action = drone_act_list[k]
                next_sta = next_state[k, :]
                reward_ind = reward[k]
                # Info = (state, action, next_sta, next_act, reward_ind )
                info = [state, action, next_sta, reward_ind]
                UAV_OB[k].qlearning(info)
                
            # Calculation of the total episodic reward of all the UAVs 
            # Calculation of the total number of connected User for the combination of all UAVs
            ##########################
            ####   Custom logs    ####
            ##########################
            episode_reward[i_episode] += sum(reward)
            episode_user_connected[i_episode] += sum(temp_data[4])
            user_connected = temp_data[4]
            
            # Also calculting episodic reward for each agent // Add this in your main program 
            episode_reward_agent = np.add(episode_reward_agent, reward)

            states = next_state
            
            # If done break from the loop (go to next episode)
            # if done:
            #     break

        #############################
        ####   Tensorboard logs  ####
        #############################
        # Track the same information regarding the performance in tensorboard log directory 
        writer.add_scalar("charts/episodic_reward", episode_reward[i_episode], i_episode)
        writer.add_scalar("charts/episodic_length", num_epochs, i_episode)
        writer.add_scalar("charts/connected_users", episode_user_connected[i_episode], i_episode)
        if args.wandb_track:
            wandb.log({"episodic_reward": episode_reward[i_episode], "episodic_length": num_epochs, "connected_users":episode_user_connected[i_episode], "global_steps": global_step})
            # wandb.log({"reward: "+ str(agent): reward[agent] for agent in range(NUM_UAV)})
            # wandb.log({"connected_users: "+ str(agent_l): user_connected[agent_l] for agent_l in range(NUM_UAV)})
        global_step += 1
        
        # Keep track of hyper parameter and other valuable information in tensorboard log directory 
        # Track the params of all agent
        # Since all agents are identical only tracking one agents params
        # writer.add_scalar("params/learning_rate", UAV_OB[1].learning_rate, i_episode )
        # writer.add_scalar("params/epsilon", UAV_OB[1].epsilon_thres, i_episode)


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
                temp_data = u_env.step(drone_act_list, args.info_exchange_lvl)
                states = u_env.get_state()
                if best_result < temp_data[4]:
                    best_result = temp_data[4]
                    best_state = states
            u_env.render(ax1)
            plt.title("Intermediate state of UAV in current episode")
            print(drone_act_list)
            print("Number of user connected in ",i_episode," episode is: ", temp_data[4])
        
        # print(epsilon)
        if epsilon > min_epsilon:
            epsilon = epsilon * decay_constant
        print(epsilon)

    return Q, episode_reward, states, temp_data[4], best_state, best_result

    #############################
    ####   Tensorboard logs  ####
    #############################
    writer.add_figure("images/uav_users_best", fig_4)
    writer.add_text(
            "best outcome", str(best_state))
    writer.add_text(
            "best result", str(best_result))
    wandb.finish()
    writer.close()

# Defining System Parameters
u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 400
num_epochs = 100
discount_factor = 0.95
alpha = 0.5
epsilon_con = 1
epsilon = 1
min_epsilon = 0.1

random.seed(10)

Q, episode_rewards, state, reward, best_state, best_result = Q_Learning(u_env, num_episode, num_epochs, discount_factor, alpha, epsilon, min_epsilon)

mdict = {'Q': Q}
savemat('Q.mat', mdict)
print(state)
print('Total Connected User in Final Stage', reward)

# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Episode vs Episodic Reward")
plt.show()
fig = plt.figure()
smoothed = smooth(episode_rewards, 10)
plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Smoothed Episode vs Episodic Reward")
plt.show()
fig = plt.figure()
final_render(state, "final")
fig = plt.figure()
final_render(best_state, "best")
print("Best State")
print(best_state)
print("The total number of connected user (as per best result)", best_result)
