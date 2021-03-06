import gym
import torch

from DDQN import Plan_RL_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from experience_replay_buf import experienceReplayBuffer

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer(burn_in=1000)
env = gym.make("Acrobot-v1")

ddqn = Plan_RL_agent(env=env, buffer=er_buf, batch_size=128, load_models= False)
ddqn.train()

'''
##### TEST monitor replanning
sum = 0
N = 10
for i in range(N):
   sum += ddqn.monitor_replanning(horizon=4)
print("media su {} esperimenti= {}".format(N, sum/N))
'''