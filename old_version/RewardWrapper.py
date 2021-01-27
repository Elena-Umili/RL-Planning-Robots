import gym
import torch
import numpy as np
import time


class RewardWrapperEncFisso(gym.Wrapper):
    def __init__(self, env, goalG, encoderPath):
        super().__init__(env)
        self.env = env

        self.goal = goalG
        self.goal = np.array(self.goal)
        self.encoder = torch.load(encoderPath)
        self.encoder.eval()

        self.goalCode = self.encoder(torch.Tensor(self.goal).float(), 50, 50)
        self.distance = self.goalCode.subtract
        self.stepCount = 0
        self.maxReward = 0


    def is_goalReached(self, state, goal):
        done = True
        for i in range (len(state)):
            if(abs(state[i] - goal[i]) > 0.2):
                done = False
                break
        return done
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state_t = torch.Tensor(next_state).float()
        # print("state: ", next_state)
        # print("goal: ", self.goal)

        my_done = self.is_goalReached(next_state, self.goal)

        # print("done :", done)
        code = self.encoder(torch.Tensor(next_state_t).float(), 50, 50)
        #print("Next_state",next_state)

        #print("Distance", torch.abs(self.distance(code)).sum().item())
        d1 = 0
        for i in range(len(next_state)):
            d1 += abs(next_state[i] - self.goal[i])

        reward_c =  10 - d1

        '''
        if reward_c >= self.maxReward:
            reward = reward_c
            self.maxReward = reward_c
        else:
            reward = -1
        '''
        reward = reward_c * 0.1

        #print(reward)

        if not my_done:
            self.stepCount += 1

        else:
            self.stepCount = 0
            reward += 200 #<-- per lunarlander
            #reward += 1000  # <-- per maze
            print("done with ", next_state)
            # next_state = np.array([0,0])
            self.maxReward = 0
            done = my_done


        return next_state, reward, done, info


class RewardWrapperEncVariabile(gym.Wrapper):
    def __init__(self, env, goalG, encoderPath):
        super().__init__(env)
        self.env = env

        self.goal = goalG
        self.goal = np.array(self.goal)
        self.encoder = torch.load(encoderPath)
        self.encoder.eval()
        self.distance = torch.nn.L1Loss()

        self.maxReward = 0
        self.stepCount = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        if (next_state[0] == self.goal[0] and next_state[1] == self.goal[1]):
            done = True
        else:
            done = False

        next_state_t = torch.Tensor(next_state).float()
        code = self.encoder(torch.Tensor(next_state_t).float(), 50, 50)
        goal_code = self.encoder(torch.Tensor(self.goal).float(), 50,
                                 50)  # <-----------------è l'unica istruzione che cambia
        reward_c = 1 - self.distance(code, goal_code).item()

        if reward_c > self.maxReward:
            reward = reward_c
            self.maxReward = reward_c
        else:
            reward = -1 / 25

        if not done:
            self.stepCount += 1
        else:
            self.stepCount = 0
            reward = 1000
            print("done with ", next_state)
            # self.maxReward = 0

        if self.stepCount >= 8 * 25:  # <-----settato per labirinti 5x5
            done = True
            # self.maxReward = 0
            self.stepCount = 0
            print("done TIMEOUT")

        return next_state, reward, done, info

# env = MazeRewardWrapperEncFisso(gym.make("LunarLander-v2"), [0,100,0,0,0,0,0,0], "planner_system/models/lunarlander/encoder")