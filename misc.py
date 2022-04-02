import gym
from gym import spaces
import numpy as np
import math
import numpy.random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class UAVenv(gym.Env):
    """Custom Environment that follows gym interface """
    metadata = {'render.modes': ['human']}

    # Fixed Input Parameters
    NUM_USER = 100              # Number of ground user
    NUM_UAV = 5                 # Number of UAV
    BS_TILT = 0                 # Angle of Antenna Tilt []
    Fc = 2                      # Operating Frequency 2 GHz
    LightSpeed = 3 * (10 ** 8)  # Speed of Light
    WaveLength = LightSpeed / (Fc * (10 ** 9))  # Wavelength of the wave
    COVERAGE_XY = 1000
    UAV_HEIGHT = 25
    UAV_VELOCITY = 10           # m/sec
    BS_LOC = np.zeros((NUM_UAV, 3))
    THETA = 2 * math.pi/NUM_UAV
    BW_RB = 180e3               # Bandwidth for a resource block
    BW_UAV = 5e6                # Total Bandwidth per UAV
    ACTUAL_BW_UAV = BW_UAV * 0.9 / BW_RB

    # UAV Base Station Distribution
    # Random Uniform Distribution of the Base Station
    # UAV Base Station Locations // Select a predefined location
    # BS_LOC = np.random.uniform(low=0.0, high=1000.0, size=(NUM_UAV, 3))
    # Set the height of the UAV randomly // For this particular keeping constant height
    # BS_LOC[:, 2] = np.random.uniform(low=25.0, high=100.0)
    BS_LOC[:, 2] = UAV_HEIGHT  # Setting heights of every UAV as 25m
    # Distance kept so as keep no overlapping cells
    BS_LOC[:, 0:2] = np.array([[250, 250], [750, 750], [250, 750], [750, 250]])
    # np.savetxt('BSLocation.txt', BS_LOC, fmt='%.3e', delimiter=' ', newline='\n')
    print(BS_LOC)

    # Load custom cellular environment describing users location
    # Square Area of 1 km x 1 km and uniformly distributed users
    # Unit of distance is given in meters

    # USER_LOC = np.random.uniform(low=0.0, high=1000.0, size=(NUM_USER, 2))
    # np.savetxt('UserLocation.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')
    USER_LOC = np.loadtxt('UserLocation.txt', dtype=np.float32, delimiter=' ')
    print(USER_LOC)
    plt.scatter(USER_LOC[:, 0], USER_LOC[:, 1])
    plt.scatter(BS_LOC[:, 0], BS_LOC[:, 1])

    plt.show()

    def __init__(self):
        super(UAVenv, self).__init__()
        # Defining action spaces // Assignment of the resource block sub-carriers
        self.action_space = spaces.Box(low=0,
                                       high=1, shape=(1, 10), dtype=np.float32)
        # Defining Observation spaces // Throughput to the users
        self.observation_space = spaces.Box(low=0,
                                            high=1, shape=(1, 15), dtype=np.float32)
        self.u_loc = self.USER_LOC
        self.bs_loc = self.BS_LOC
        self.state = np.zeros(self.NUM_UAV, 3)
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA/2)

    def step(self, action):
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        quit_in = self.state[:, 2] <= 0

        # Calculate the distance of every users to the UAV BS and organize as a list
        dist_u_uav = np.array([])
        temp_dist = []
        for i in range(self.NUM_UAV):
            for j in range(self.NUM_USER):
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

## Last test
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
    NUM_USER = 100              # Number of ground user
    NUM_UAV = 5                 # Number of UAV
    BS_TILT = 0                 # Angle of Antenna Tilt []
    Fc = 2                      # Operating Frequency 2 GHz
    LightSpeed = 3 * (10 ** 8)  # Speed of Light
    WaveLength = LightSpeed / (Fc * (10 ** 9))  # Wavelength of the wave
    COVERAGE_XY = 1000
    UAV_HEIGHT = 25
    UAV_VELOCITY = 10           # m/sec
    BS_LOC = np.zeros((NUM_UAV, 3))
    THETA = 15 * math.pi/180    # in radian
    BW_RB = 180e3               # Bandwidth for a resource block
    BW_UAV = 5e6                # Total Bandwidth per UAV
    ACTUAL_BW_UAV = BW_UAV * 0.9 / BW_RB

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
        self.action_space = spaces.discrete(shape=(1, self.NUM_UAV), dtype=np.integer32)
        # Defining Observation spaces // UAV RB to each user
        # Position of the UAV in space // constant height and X and Y pos
        self.observation_space = spaces.discrete(shape=(2, NUM_UAV), dtype=np.integer32)
        self.u_loc = self.USER_LOC
        self.state = np.zeros((self.NUM_UAV, 3))
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA / 2)

    def step(self, action):
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        quit_in = self.state[:, 2] <= 0

        # Calculate the distance of every users to the UAV BS and organize as a list
        dist_u_uav = np.array([])
        temp_dist = []
        for i in range(self.NUM_UAV):
            tem_x = self.state[i][0]
            tem_y = self.state[i][1]
            # one step action
            self.state[i][0]
            self.state[i][1]
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
