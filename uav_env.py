import gym
from gym import spaces
import numpy as np
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class UAVenv(gym.Env):
    """Custom Environment that follows gym interface """
    metadata = {'render.modes': ['human']}

    # Fixed Input Parameters
    NUM_USER = 100  # Number of ground user
    NUM_UAV = 5  # Number of UAV
    BS_TILT = 0  # Angle of Antenna Tilt []
    Fc = 2  # Operating Frequency 2 GHz
    LightSpeed = 3 * (10 ** 8)  # Speed of Light
    WaveLength = LightSpeed / (Fc * (10 ** 9))  # Wavelength of the wave
    COVERAGE_XY = 1000
    UAV_HEIGHT = 25
    UAV_VELOCITY = 10  # m/sec
    BS_LOC = np.zeros((NUM_UAV, 3))
    THETA =  60 * math.pi / 180  # in radian    BW_RB = 180e3  # Bandwidth for a resource block
    BW_UAV = 5e6  # Total Bandwidth per UAV
    BW_RB = 180e3 # Bandwidth of a Resource Block
    ACTUAL_BW_UAV = BW_UAV * 0.9 / BW_RB
    GRID_SIZE = COVERAGE_XY / 10  # Each grid defined as 100m block

    # User distribution on the target area // NUM_USER/5 users in each of four hotspots
    # Remaining NUM_USER/5 is then uniformly distributed in the target area

    # HOTSPOTS = np.array([[200, 200], [800, 800], [300, 800], [800, 300]])
    # USER_DIS = int(NUM_USER / NUM_UAV)
    # USER_LOC = np.zeros((NUM_USER-USER_DIS, 2))
    #
    # for i in range(len(HOTSPOTS)):
    #     for j in range(USER_DIS):
    #         temp_loc_1 = random.uniform(HOTSPOTS[i, 0]-100, HOTSPOTS[i, 0]+100)
    #         temp_loc_2 = random.uniform(HOTSPOTS[i, 1]-100, HOTSPOTS[i, 1]+100)
    #         USER_LOC[i*USER_DIS+j, :] = [temp_loc_1, temp_loc_2]
    # temp_loc = np.random.uniform(low=0.0, high=COVERAGE_XY, size=(USER_DIS, 2))
    # USER_LOC = np.concatenate((USER_LOC, temp_loc))
    # np.savetxt('UserLocation.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')

    # Saving the user location on a file instead of generating everytime

    USER_LOC = np.loadtxt('UserLocation.txt', dtype=np.float32, delimiter=' ')
    plt.scatter(USER_LOC[:, 0], USER_LOC[:, 1])
    plt.show()

    def __init__(self):
        super(UAVenv, self).__init__()
        # Defining action spaces // UAV RB allocation to each user increase each by 1 until remains
        # Five different action for the movement of each UAV
        # 1 = Right, 2 = Left, 3 = straight, 4 = back ,5 = Hover
        self.action_space = spaces.Discrete(5)
        # Defining Observation spaces // UAV RB to each user
        # Position of the UAV in space // constant height and X and Y pos
        self.observation_space = spaces.Discrete(self.NUM_UAV)
        self.u_loc = self.USER_LOC
        self.state = np.zeros((self.NUM_UAV, 3), dtype=np.int32)
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA / 2)

    def step(self, action):
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        flag = 0
        # Calculate the distance of every users to the UAV BS and organize as a list
        dist_u_uav = np.array([])
        temp_dist = []
        for i in range(self.NUM_UAV):
            tem_x = self.state[i, 0]
            tem_y = self.state[i, 1]
            # one step action
            if action[i] == 1:
                self.state[0, i] = self.state[0, i] + 1
            elif action[i] == 2:
                self.state[0, i] = self.state[0, i] - 1
            elif action[i] == 3:
                self.state[1, i] = self.state[1, i] + 1
            elif action[i] == 4:
                self.state[i, 2] = self.state[1, i] - 1
            elif action[i] == 5:
                pass
            else:
                print("Error Action Value")

            # Take boundary condition into account
            if self.state[0, i] < 0 or self.state[0, i] > self.GRID_SIZE or self.state[1, i] < 0 or self.state[1, i] > \
                    self.GRID_SIZE:
                self.state[0, i] = tem_x
                self.state[1, i] = tem_y
                flag += 1  # Later punish in reward function

        u_loc = self.USER_LOC
        # Calculation of the distance value for all UAV and User
        for k in range(self.NUM_UAV):
            for l in range(self.NUM_USER):
                dist_u_uav[k, l] = math.sqrt((u_loc[l, 0] - self.state[0, k]) ** 2 + (u_loc[l, 1] -
                                                                                      self.state[1, k]) ** 2)
        max_user_num = self.ACTUAL_BW_UAV / self.BW_RB

        # User association to the UAV based on the distance value.
        # First do a single sweep by all the Users to request to connect to the closest UAV
        # After the first sweep is complete the UAV will admit a certain Number of Users based on available resource
        # In the second sweep the User will request to the UAV that is closest to it and UAV will admit the User if resource available
        user_uav_list = np.array([])
        for i in range(self.NUM_USER):
            temp_ind = np.where(dist_u_uav[i,:] >= self.coverage_radius)
            # Padding of the rest of the value with 0 to remove dimensional error
            np.pad(temp_ind, (0, 5-len(temp_ind)%5), 'constant')
            dist = dist_u_uav[i, temp_ind]
            # Sort the distance value and arrange the index
            dist[::-1].sort()
            temp_ind = np.where(dist_u_uav[i,:] == dist)
            # Index of the list represents UAV where as value represent distance
            user_uav_list = np.concatenate([user_uav_list, [i, temp_ind]], axis=0)

        user_uav_row, user_uav_col = user_uav_list.shape

        # Associate the user to UAV
        uav_asso = np.zeros((self.NUM_UAV,2), dtype="int")
        user_asso = np.zeros((self.NUM_USER,2), dtype="int")
        for j in range(0,5):
            for i in range(self.NUM_UAV):
                t_idx = np.where(user_uav_list[:,1] == i)
                for k in range(t_idx):
                    if uav_asso[i,1] <= max_user_num:
                        uav_asso[i, 1] += 1










        # User association to the UAV based on the SINR value unless full
        # First preparation of SINR value for UAV User pair inside the coverage range

        user_assm = np.zeros((self.NUM_UAV, 1), dtype=np.int32)

        for i in range(self.NUM_UAV):
             if user_assm[i] <= max_user_num:

            # Distance formula ( UAV loc is stored as (X,Y,Z))
            temp_dist[j] = math.sqrt(
                (self.u_loc[j, 0] - self.bs_loc[i, 0]) ** 2 + (self.u_loc[j, 1] - self.bs_loc[i,
                                                                                              1]) ** 2)
            dist_u_uav[:, 0:2] = np.insert(dist_u_uav, [np.argmin(temp_dist), i], axis=1)

        # Compare if the closest UAV User distance is within that's BS (UAV) coverage area
        plane_distance = np.sqrt(np.dot(dist_u_uav, dist_u_uav) - np.dot(self.UAV_HEIGHT, self.UAV_HEIGHT))
        if plane_distance <= self.coverage_radius:
            pass

    def render(self):
        # implement viz
        pass

    def reset(self):
        # reset out states
        pass
