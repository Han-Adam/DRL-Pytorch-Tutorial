import gym
import numpy as np
from gym import spaces

# change the observation space from [21, 20, 3] to [3, 20, 21].
# aims to change the channel to the first axis.
# accord with pytorch format.

class Channel_switch(gym.ObservationWrapper):
    def __init__(self, env):
        super(Channel_switch, self).__init__(env)
        original_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape= [original_shape[-1],
                                                        original_shape[1],
                                                        original_shape[0]])

    def observation(self, observation):
        return np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0)


def env_wrap(env):
    return Channel_switch(env)