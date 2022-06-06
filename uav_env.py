
###################################
## Environment Setup of for UAV  ##
###################################

import gym
from gym import spaces
import numpy as np
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random


###################################
##     UAV Class Defination      ##
###################################

class UAVenv(gym.Env):
    """Custom Environment that follows gym interface """
    metadata = {'render.modes': ['human']}

    # Fixed Input Parameters
    NUM_USER = 100  # Number of ground user
    NUM_UAV = 5  # Number of UAV
    Fc = 2  # Operating Frequency 2 GHz
    LightSpeed = 3 * (10 ** 8)  # Speed of Light
    WaveLength = LightSpeed / (Fc * (10 ** 9))  # Wavelength of the wave
    COVERAGE_XY = 1000
    UAV_HEIGHT = 300
    BS_LOC = np.zeros((NUM_UAV, 3))
    THETA = 35 * math.pi / 180  # in radian   # Bandwidth for a resource block (This value is representing 2*theta instead of theta)
    BW_UAV = 5e6  # Total Bandwidth per UAV
    BW_RB = 180e3  # Bandwidth of a Resource Block
    ACTUAL_BW_UAV = BW_UAV * 0.9
    grid_space = 100
    GRID_SIZE = int(COVERAGE_XY / grid_space)  # Each grid defined as 100m block

    # User distribution on the target area // NUM_USER/5 users in each of four hotspots
    # Remaining NUM_USER/5 is then uniformly distributed in the target area

    # HOTSPOTS = np.array(
    #     [[200, 200], [800, 800], [300, 800], [800, 300]])  # Position setup in grid size rather than actual distance
    # USER_DIS = int(NUM_USER / NUM_UAV)
    # USER_LOC = np.zeros((NUM_USER - USER_DIS, 2))
    
    # for i in range(len(HOTSPOTS)):
    #     for j in range(USER_DIS):
    #         temp_loc_1 = random.uniform(HOTSPOTS[i, 0] - 100, HOTSPOTS[i, 0] + 100)
    #         temp_loc_2 = random.uniform(HOTSPOTS[i, 1] - 100, HOTSPOTS[i, 1] + 100)
    #         USER_LOC[i * USER_DIS + j, :] = [temp_loc_1, temp_loc_2]
    # temp_loc = np.random.uniform(low=0, high=COVERAGE_XY, size=(USER_DIS, 2))
    # USER_LOC = np.concatenate((USER_LOC, temp_loc))
    # np.savetxt('UserLocation.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')

    # Saving the user location on a file instead of generating everytime

    USER_LOC = np.loadtxt('UserLocation.txt', dtype=np.int32, delimiter=' ')

    # plt.scatter(USER_LOC[:, 0], USER_LOC[:, 1])
    # plt.show()


    def __init__(self):
        super(UAVenv, self).__init__()
        # Defining action spaces // UAV RB allocation to each user increase each by 1 until remains
        # Five different action for the movement of each UAV
        # 1 = Right, 2 = Left, 3 = straight, 4 = back ,5 = Hover
        self.action_space = spaces.Discrete(5)
        # Defining Observation spaces // UAV RB to each user
        # Position of the UAV in space // constant height and X and Y pos
        # self.observation_space = spaces.Discrete(self.NUM_UAV)
        self.u_loc = self.USER_LOC
        self.state = np.zeros((self.NUM_UAV, 3), dtype=np.int32)
        # set the states to the hotspots and one at the centre for faster convergence
        # further complexity by choosing random value of state
        self.state[:, 0:2] = [[2, 2], [8, 5], [4, 7], [10, 3], [2, 3]]
        self.state[:, 2] = self.UAV_HEIGHT
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA / 2)
        self.coverage_area = 2*self.coverage_radius
        print(self.coverage_area)

    def step(self, action):
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        flag = 0
        # Calculate the distance of every users to the UAV BS and organize as a list
        dist_u_uav = np.zeros(shape=(self.NUM_UAV, self.NUM_USER))
        for i in range(self.NUM_UAV):
            tem_x = self.state[i, 0]
            tem_y = self.state[i, 1]
            # one step action
            if action[i] == 1:
                self.state[i, 0] = self.state[i, 0] + 1
            elif action[i] == 2:
                self.state[i, 0] = self.state[i, 0] - 1
            elif action[i] == 3:
                self.state[i, 1] = self.state[i, 1] + 1
            elif action[i] == 4:
                self.state[i, 1] = self.state[i, 1] - 1
            elif action[i] == 5:
                pass
            else:
                print("Error Action Value")

            # Take boundary condition into account
            if self.state[i, 0] < 0 or self.state[i, 0] > self.GRID_SIZE or self.state[i, 1] < 0 or self.state[i, 1] > \
                    self.GRID_SIZE:
                self.state[i, 0] = tem_x
                self.state[i, 1] = tem_y
                flag += 1  # Later punish in reward function

        # Calculation of the distance value for all UAV and User
        for k in range(self.NUM_UAV):
            for l in range(self.NUM_USER):
                dist_u_uav[k, l] = math.sqrt((self.u_loc[l, 0] - (self.state[k, 0] * self.grid_space)) ** 2 + (self.u_loc[l, 1] -
                                                                                      (self.state[k, 1] * self.grid_space)) ** 2)
        max_user_num = self.ACTUAL_BW_UAV / self.BW_RB

        ######################
        ## Final Algorithm  ##
        ######################

        # User association to the UAV based on the distance value. First do a single sweep by all
        # the Users to request to connect to the closest UAV After the first sweep is complete the UAV will admit a
        # certain Number of Users based on available resource In the second sweep the User will request to the UAV
        # that is closest to it and UAV will admit the User if resource available

        # Connection request is a np array matrix that contains UAV Number as row and
        # User Number Connected to it on Columns and is stored in individual UAV to keep track of the
        # User requesting to connect

        connection_request = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")

        for i in range(self.NUM_USER):
            if not(np.any(connection_request[:,i] == 1)):                      # Skip if connection request already sent
                close_uav = np.argmin(dist_u_uav[:,i])                    # Closest UAV index
                if dist_u_uav[close_uav, i] <= self.coverage_area:        # UAV - User distance within the coverage radius then only connection request
                    connection_request[close_uav, i] = 1                  # All staifies, then connection request for the UAV - User

        # print(connection_request)

        # Allocating only 70% of max cap in first run
        # After all the user has send their connection request,
        # UAV only admits Users closest to and if bandwidth is available
        user_asso_flag = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")
        for i in range(self.NUM_UAV):
            # Maximum Capacity for a single UAV
            cap_user_num = int(0.7* max_user_num)
            # Sorting the users with the connection request to this UAV
            temp_user = np.where(connection_request[i, :] == 1)
            temp_user_distance = dist_u_uav[i, temp_user]
            # print(temp_user)
            # print(temp_user_distance)
            temp_user_sorted = np.argsort(temp_user_distance) # Contains user index with closest 2D distance value (out of requested user)
            # The user list are already sorted, to associate flag bit of user upto the index from
            # min(max_user, max_number_of_user_inside_coverage_area)
            temp_user_idx = temp_user_sorted[0, 0:min(cap_user_num, (np.size(temp_user_sorted)))]
            temp_user_actual_idx = np.where(dist_u_uav[i,:] == temp_user_distance[0,temp_user_idx])
            # Set user association flag to 1 for that UAV and closest user index

            user_asso_flag[i, temp_user_actual_idx] = 1
            # print(temp_user_actual_idx)
            # print(user_asso_flag)


        # print(user_asso_flag)
        # For the second sweep, sweep through all users
        # If the user is not associated choose the closest UAV and check whether it has any available resource
        # If so allocate the resource and set the User association flag bit of that user to 1
        for j in range(self.NUM_USER):
            if np.any(user_asso_flag[:, j] != 0):
                close_uav_id = dist_u_uav[:, j]
                close_uav_id = [i[0] for i in sorted(enumerate(close_uav_id), key=lambda x: x[1])]
                for close_id in close_uav_id:
                    if np.sum(user_asso_flag[close_id,:]) <= max_user_num:
                        user_asso_flag[close_id, j] = 1
                        break

        # print(user_asso_flag)
        # Need to work on the return parameter of done, info, reward, and obs
        # Calculation of reward function too i.e. total bandwidth provided to the user

        reward = sum(sum(user_asso_flag))
        # print(reward)

        if flag != 0:
            isDone = True

        # Return of obs, reward, done, info
        return np.copy(self.state).reshape(1, self.NUM_UAV * 3), reward, isDone, "empty"

    def render(self, ax, mode='human', close=False):
        # implement viz
        if mode == 'human':
            position = self.state[:, 0:2] * self.grid_space
            ax.scatter(self.u_loc[:, 0], self.u_loc[:, 1], c = '#ff0000', marker='o')
            ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='x')
            plt.pause(0.5)
            plt.draw()

    def reset(self):
        # reset out states
        # set the states to the hotspots and one at the centre for faster convergence
        # further complexity by choosing random value of state
        self.state[:, 0:2] = [[2, 2], [8, 5], [4, 7], [10, 3], [2, 3]]
        self.state[:, 2] = self.UAV_HEIGHT
        return self.state

    def get_state(self):
        state_loc = np.zeros((self.NUM_UAV, 2))
        for k in range(self.NUM_UAV):
            state_loc[k, 0] = self.state[k, 0]
            state_loc[k, 1] = self.state[k, 1]
        return state_loc

    def get_full_obs(self):
        obs = np.ones((self.COVERAGE_XY + 1, self.COVERAGE_XY + 1, 3), dtype=float)
        for i in range(self.NUM_USER):
            obs[self.USER_LOC[i, 0], self.USER_LOC[i, 1], 0] = 1
            obs[self.USER_LOC[i, 0], self.USER_LOC[i, 1], 1] = 0
            obs[self.USER_LOC[i, 0], self.USER_LOC[i, 1], 2] = 0
        for i in range(self.NUM_UAV):
            obs[self.state[i, 0] * self.grid_space, self.state[i, 1] * self.grid_space, 0] = 0
            obs[self.state[i, 0] * self.grid_space, self.state[i, 1] * self.grid_space, 1] = 0
            obs[self.state[i, 0] * self.grid_space, self.state[i, 1] * self.grid_space, 2] = 1
        return obs
    
    def get_drone_obs(self, state):
        obs_size = int(2 * self.coverage_radius - 1)
        obs = np.ones((obs_size, obs_size, 3))
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + state[0] - self.coverage_radius + 1
                y = j + state[1] - self.coverage_radius + 1

                for k in range(self.NUM_USER):
                    
                    if self.u_loc[k, 0] == x and self.u_loc[k, 1] == y:
                        obs[i, j, 0] = 1
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0

                    if x < 0 or x > (self.GRID_SIZE  - 1)*self.grid_space or y < 0 or y > (self.GRID_SIZE - 1)*self.grid_space:
                        obs[i, j, 0] = 0.5
                        obs[i, j, 1] = 0.5
                        obs[i, j, 2] = 0.5

                    if (self.coverage_radius - 1 - i) * (self.coverage_radius - 1 - i) + (
                            self.coverage_radius - 1 - j) * (
                            self.coverage_radius - 1 - i) > self.coverage_radius * self.coverage_radius:
                        obs[i, j, 0] = 0.5
                        obs[i, j, 1] = 0.5
                        obs[i, j, 2] = 0.5

                    return obs

    def get_joint_obs(self):
        obs = np.ones((self.GRID_SIZE, self.GRID_SIZE, 3))
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                obs[i, j, 0] = 0.5
                obs[i, j, 1] = 0.5
                obs[i, j, 2] = 0.5

        for k in range(self.NUM_UAV):
            temp = self.get_drone_obs(self.state[k,:])
            size = temp.shape[0]
            for i in range(size):
                for j in range(size):
                    x = i + self.state[k, 0] - self.coverage_radius + 1
                    y = j + self.state[k, 1] - self.coverage_radius + 1
                    if_obs = True
                    if temp[i, j, 0] == 0.5 and temp[i, j, 1] == 0.5 and temp[i, j, 2] == 0.5:
                        if_obs = False
                    if if_obs == True:
                        obs[x, y, 0] = temp[i, j, 0]
                        obs[x, y, 1] = temp[i, j, 1]
                        obs[x, y, 2] = temp[i, j, 2]

        for k in range(self.NUM_UAV):
            obs[self.state[k, 0], self.state[k, 1], 0] = 0
            obs[self.state[k, 0], self.state[k, 1], 1] = 0
            obs[self.state[k, 0]. self.state[k, 1], 2] = 1

        return obs
