import numpy as np
import gym
from gym import spaces
#from gym_duckietown.envs import DuckietownEnv

class PositionObservation(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PositionObservation, self).__init__(env)
        self.observation_space = spaces.Box(
            low=np.array([-0.5,0,-90]),             #,-np.pi/2
            high=np.array([0.5,1,90]),              #,np.pi/2
            shape=(3,),
            dtype=np.float32
        )
    def observation(self, observation):
        obs = self.get_agent_info()
        #print("internal obs = ", obs)
        obs = obs['Simulator']['lane_position']
        obs = dict((k,obs[k]) for k in ('dist', 'dot_dir', 'angle_deg'))
        obs = list(obs.values())
        return np.array(obs)

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward

TURN_VEL = 0.2
TURN_ANGLE = 1
STRAIGHT_VEL = 0.6

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)  # 3 since we need also to use it in a trained model
    
    # velocity and steering angle
    def action(self, action):
        # Turn left                         
        if action == 0:
            vels = [TURN_VEL, TURN_ANGLE]              
        # Turn right
        elif action == 1:
            vels = [TURN_VEL, -TURN_ANGLE]              
        # Go forward
        elif action == 2:
            vels = [STRAIGHT_VEL, 0.0]               
        # stationary
        elif action == 3:
            vels = [0.0, 0.0]              
        # go backwards
        elif action == 4:
            vels = [-STRAIGHT_VEL, 0.0]             
        else:
            assert False, "unknown action"
        return np.array(vels)

class DiscreteActionWrapperTrain(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    Used for training with three actions
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [TURN_VEL, TURN_ANGLE] #[0.2, +1.0]              #original: [0.6, +1.0]
        # Turn right
        elif action == 1:
            vels = [TURN_VEL, -TURN_ANGLE] #[0.2, -1.0]              #original: [0.6, -1.0]
        # Go forward
        elif action == 2:   
            vels = [STRAIGHT_VEL, 0.0] #[0.3, 0.0]               #original: [0.7, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class NoiseWrapper(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)

    def action(self, action):
        action[0] += np.random.normal(0, TURN_VEL * (1/100)) # 1% noise vel*1%
        action[1] += np.random.normal(0, TURN_ANGLE * (1/100))   # 1% noise angle*1%
        return action

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2][::-1])) #[::-1] to reverse

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
