# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
import gym_maze
import random

from Plan_RL_agent import Plan_RL_agent
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from experience_replay_buf import experienceReplayBuffer

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


    er_buf = experienceReplayBuffer()

    #random.seed(8)
    env = gym.make("Acrobot-v1")
    ddqn = Plan_RL_agent(env=env, buffer=er_buf, batch_size=64)
    ddqn.train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/