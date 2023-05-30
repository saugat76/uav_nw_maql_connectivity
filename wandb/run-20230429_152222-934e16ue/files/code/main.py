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


os.chdir = ("")

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

# Define arg parser with default values
def parse_args():
    parser = argparse.ArgumentParser()
    # Arguments for the experiments name / run / setup and Weights and Biases
    parser.add_argument("--exp-name", type=str, default="maql_uav", help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of experiment to ensure reproducibility")
    parser.add_argument("--torch-deterministic", type= lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggeled, 'torch-backends.cudnn.deterministic=False'")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-track", type=lambda x: bool(strtobool(x)), default=False, help="if toggled, this experiment will be tracked with Weights and Biases project")
    parser.add_argument("--wandb-name", type=str, default="UAV_Subband_Allocation_QL_Pytorch", help="project name in Weight and Biases")
    parser.add_argument("--wandb-entity", type=str, default= None, help="entity(team) for Weights and Biases project")

    # Arguments specific to the Algotithm used 
    parser.add_argument("--env-id", type=str, default= "ma-custom-UAV-connectivity", help="id of developed custom environment")
    parser.add_argument("--num-env", type=int, default=1, help="number of parallel environment")
    parser.add_argument("--num-episode", type=int, default=351, help="number of episode, default value till the trainning is progressed")
    parser.add_argument("--num-steps", type=int, default= 100, help="number of steps/epoch use in every episode")
    parser.add_argument("--learning-rate", type=float, default= 0.5, help="learning rate of the dql alggorithm used by every agent")
    parser.add_argument("--gamma", type=float, default= 0.95, help="discount factor used for the calculation of q-value, can prirotize future reward if kept high")
    parser.add_argument("--batch-size", type=int, default= 512, help="batch sample size used in a trainning batch")
    parser.add_argument("--epsilon", type=float, default= 0.1, help="epsilon to set the eploration vs exploitation")
    parser.add_argument("--update-rate", type=int, default= 10, help="steps at which the target network updates it's parameter from main network")
    parser.add_argument("--buffer-size", type=int, default=125000, help="size of replay buffer of each individual agent")
    parser.add_argument("--epsilon-min", type=float, default=0.1, help="maximum value of exploration-exploitation paramter, only used when epsilon deacay is set to True")
    parser.add_argument("--epsilon-decay", type=lambda x: bool(strtobool(x)), default=False, help="epsilon decay is used, explotation is prioritized at early episodes and on later epsidoe exploitation is prioritized, by default set to False")
    parser.add_argument("--epsilon-decay-steps", type=int, default=1, help="set the rate at which is the epsilon is deacyed, set value equates number of steps at which the epsilon reaches minimum")

    # Environment specific arguments 
    parser.add_argument("--info-exchange-lvl", type=int, default=1, help="information exchange level between UAVs: 1 -> implicit, 2 -> reward, 3 -> position with distance penalty, 4 -> state")
    # Arguments for used inside the wireless UAV based enviornment  
    parser.add_argument("--num-user", type=int, default=100, help="number of user in defined environment")
    parser.add_argument("--num-uav", type=int, default=5, help="number of uav for the defined environment")
    parser.add_argument("--generate-user-distribution", type=lambda x: bool(strtobool(x)), default=False, help="if true generate a new user distribution, set true if changing number of users")
    parser.add_argument("--carrier-freq", type=int, default=2, help="set the frequency of the carrier signal in GHz")
    parser.add_argument("--coverage-xy", type=int, default=1000, help="set the length of target area (square)")
    parser.add_argument("--uav-height", type=int, default=350, help="define the altitude for all uav")
    parser.add_argument("--theta", type=int, default=60, help="angle of coverage for a uav in degree")
    parser.add_argument("--bw-uav", type=float, default=4e6, help="actual bandwidth of the uav")
    parser.add_argument("--bw-rb", type=float, default=180e3, help="bandwidth of a resource block")
    parser.add_argument("--grid-space", type=int, default=100, help="seperating space for grid")
    parser.add_argument("--uav-dis-th", type=int, default=1000, help="distance value that defines which uav agent share info")
    parser.add_argument("--dist-pri-param", type=float, default=1/5, help="distance penalty priority parameter used in level 3 info exchange")
    
    args = parser.parse_args()

    return args


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
        self.Q = np.random.rand(self.state_space_size[0] + 1, self.state_space_size[1]+ 1, self.action_space_size)

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

if __name__ == "__main__":
    args = parse_args()
    u_env = UAVenv(args)
    GRID_SIZE = u_env.GRID_SIZE
    NUM_UAV = u_env.NUM_UAV
    NUM_USER = u_env.NUM_USER
    num_episode = args.num_episode
    num_epochs = args.num_steps
    discount_factor = args.gamma
    alpha = args.learning_rate
    batch_size = args.batch_size
    update_rate = args.update_rate
    epsilon = args.epsilon
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay_steps
    dnn_epoch = 1

    # Set the run id name to tack all the runs 
    run_id = f"{args.exp_name}__lvl{args.info_exchange_lvl}__{u_env.NUM_UAV}__{args.seed}__{int(time.time())}"

    ## The whole problem of no convergence and no oscillation was dues to really low learning rate, as a result it was working 
    ## really slowly as a result it was taking way to long to converge to more optimal position
    ## Having a higher learning rate helped in this case 

    # Set the seed value from arg parser to ensure reproducibility 
    random.seed(args.seed)
    np.random.seed(args.seed)

    # If wandb tack is set to True // Track the training process, hyperparamters and results
    if args.wandb_track:
        wandb.init(
            # Set the wandb project where this run will be logged
            project=args.wandb_name,
            entity=args.wandb_entity,
            sync_tensorboard= True,
            # track hyperparameters and run metadata
            config=vars(args),
            name= run_id,
            save_code= True,
        )
    # Track everyruns inside run folder // Tensorboard files to keep track of the results
    writer = SummaryWriter(f"runs/{run_id}")
    # Store the hyper paramters used in run as a Scaler text inside the tensor board summary
    writer.add_text(
        "hyperparamaters", 
        "|params|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()]))
    )

    # Store environment specific parameters
    env_params = {'num_uav':NUM_UAV, 'num_user': NUM_USER, 'grid_size': GRID_SIZE, 'start_pos': str(u_env.state), 
                      'coverage_xy':u_env.COVERAGE_XY, 'uav_height': u_env.UAV_HEIGHT, 'bw_uav': u_env.BW_UAV, 
                      'bw_rb':u_env.BW_RB, 'actual_bw_uav':u_env.ACTUAL_BW_UAV, 'uav_dis_thres': u_env.UAV_DIST_THRS,
                      'dist_penalty_pri': u_env.dis_penalty_pri}
    writer.add_text(
        "environment paramters", 
        "|params|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in env_params.items()]))
    )

    # Initialize global step value
    global_step = 0

    # Keeping track of the episode reward
    episode_reward = np.zeros(num_episode)
    episode_user_connected = np.zeros(num_episode)

    # Keeping track of individual agents 
    episode_reward_agent = np.zeros((NUM_UAV, 1))


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

            # Find the rewards for the combined set of actions for the UAV
            temp_data = u_env.step(drone_act_list, args.info_exchange_lvl)
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
                states_fin = states
                if best_result < sum(temp_data[4]):
                    best_result = sum(temp_data[4])
                    best_state = states
            
            # Custom logs and figures save / 
            custom_dir = f'custom_logs\lvl_{args.info_exchange_lvl}\{run_id}'
            if not os.path.exists(custom_dir):
                os.makedirs(custom_dir)
            
            u_env.render(ax1)
            plt.title("Simulation")
            #############################
            ####   Tensorboard logs  ####
            #############################
            # writer.add_figure("images/uav_users", figure, i_episode)
            writer.add_scalar("charts/connected_users_test", sum(temp_data[4]))

            print(drone_act_list)
            print("Number of user connected in ",i_episode," episode is: ", temp_data[4])
            print("Total user connected in ",i_episode," episode is: ", sum(temp_data[4]))
    
    def smooth(y, pts):
        box = np.ones(pts)/pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    ##########################
    ####   Custom logs    ####
    ##########################
    ## Save the data from the run as a file in custom logs
    mdict = {'num_episode':range(0, num_episode),'episodic_reward': episode_reward}
    savemat(custom_dir + f'\episodic_reward.mat', mdict)
    mdict_2 = {'num_episode':range(0, num_episode),'connected_user': episode_user_connected}
    savemat(custom_dir + f'\connected_users.mat', mdict_2)
    mdict_3 = {'num_episode':range(0, num_episode),'episodic_reward_agent': episode_reward_agent}
    savemat(custom_dir + f'\epsiodic_reward_agent.mat', mdict_3)
    
    # Plot the accumulated reward vs episodes // Save the figures in the respective directory 
    # Episodic Reward vs Episodes
    fig_1 = plt.figure()
    plt.plot(range(0, num_episode), episode_reward)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Episode vs Episodic Reward")
    plt.savefig(custom_dir + f'\episode_vs_reward.png')
    plt.close()
    # Episode vs Connected Users
    fig_2 = plt.figure()
    plt.plot(range(0, num_episode), episode_user_connected)
    plt.xlabel("Episode")
    plt.ylabel("Connected User in Episode")
    plt.title("Episode vs Connected User in Episode")
    plt.savefig(custom_dir + f'\episode_vs_connected_users.png')
    plt.close()
    # Episodic Reward vs Episodes (Smoothed)
    fig_3 = plt.figure()
    smoothed = smooth(episode_reward, 10)
    plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Smoothed Episode vs Episodic Reward")
    plt.savefig(custom_dir + f'\episode_vs_rewards(smoothed).png')
    plt.close()
    # Plot for best and final states 
    fig = plt.figure()
    final_render(states_fin, "final")
    plt.savefig(custom_dir + r'\final_users.png')
    plt.close()
    fig_4 = plt.figure()
    final_render(best_state, "best")
    plt.savefig(custom_dir + r'\best_users.png')
    plt.close()
    print(states_fin)
    print('Total Connected User in Final Stage', temp_data[4])
    print("Best State")
    print(best_state)
    print("Total Connected User (Best Outcome)", best_result)

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

