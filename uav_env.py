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
    THETA = 60 * math.pi / 180  # in radian    BW_RB = 180e3  # Bandwidth for a resource block
    BW_UAV = 5e6  # Total Bandwidth per UAV
    BW_RB = 180e3  # Bandwidth of a Resource Block
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

        # # Final Algorithm User association to the UAV based on the distance value. First do a single sweep by all
        # the Users to request to connect to the closest UAV After the first sweep is complete the UAV will admit a
        # certain Number of Users based on available resource In the second sweep the User will request to the UAV
        # that is closest to it and UAV will admit the User if resource available

        # Connection request is a np array matrix that contains UAV Number as row and
        # User Number Connected to it on Columns and is stored in individual UAV to keep track of the
        # User requesting to connect

        connection_request = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")
        for i in range(self.NUM_UAV):
            for j in range(self.NUM_USER):
                idx = 0
                if dist_u_uav[i, j] <= self.coverage_radius:
                    connection_request[i, idx] = j + 1  # Increasing the user number by 1 to not confuse with empty val
                    idx += 1

        # Allocating only 70% of max cap in first run
        cap_user_num = int(0.7 * max_user_num)
        # After all the user has send their connection request,
        # UAV only admits Users closest to and if bandwidth is available
        user_asso_flag = np.zeros(shape=(self.NUM_USER, 2), dtype="int")
        uav_asso = []
        for i in range(self.NUM_UAV):
            distance_list = {}
            for j in list(connection_request[i, :]):
                if j != 0:
                    # Dict first value is Users_ID and Second value is the distance parameter
                    # Subtract 1 as the connection request has index of User + 1
                    distance_list.update({j - 1, dist_u_uav[i, j - 1]})
            dict(sorted(distance_list.items(), key=lambda item: item[1]))
            # Make list of user sorted based on distance value
            distance_user_list = list(distance_list)
            # Remove the users from dict outside coverage area
            distance_list = {key: val for key, val in distance_list.items() if val > 0}
            # Select user index with min value of distance
            min_dist_id = min(distance_list, key=distance_list.get)
            # The user list are already sorted, to associate flag bit of user upto the index from
            # min(max_user, max_number_of_user_inside_coverage_area)
            distance_user_list = distance_user_list[0:min(cap_user_num, min_dist_id)]
            # If the user have been allocated the resource set the user association flag bit to 1
            # It can be thought of as the user denoting it self connected
            user_asso_flag[distance_user_list, 0] = 1
            # Still need to take multi-UAV coverage to a single UAV
            uav_asso[i] += min(max_user_num, min_dist_id)

        # For the second sweep, sweep through all users
        # If the user is not associated choose the closest UAV and check whether it has any available resource
        # If so allocate the resource and set the User association flag bit of that user to 1
        for j in range(self.NUM_USER):
            if user_asso_flag[j, 0] != 0:
                close_uav_id = np.argmin(dist_u_uav[:, j])
                if uav_asso[close_uav_id] <= max_user_num:
                    uav_asso[close_uav_id] += 1
                    user_asso_flag[j, 0] = 1

    def render(self):
        # implement viz
        pass

    def reset(self):
        # reset out states
        pass
