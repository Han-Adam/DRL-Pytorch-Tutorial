'''
This is a single agent demo of  DeepMind's <Proximal Policy Optimization Algorithms>
Compared with PPO1, this algorithm contains GAE, Batch Update.
'''
import gym
from PPO import PPO
import os
import yaml
import numpy as np
import torch
from PPO import PPO
import os

import matplotlib.pyplot as plt


class GYM:
    def __init__(self):
        self.s = None
        self.r = None
        self.done = None

        self.M = 1
        self.G = 9.8
        self.f = 0.1
        self.F_mid = 10
        self.t = 0.1

    def reset(self):
        self.s = None
        self.r = None
        self.done = None

        position = 0
        velocity = 0
        acceleration = 0
        target = np.random.rand()*40. + 30.

        self.s = [position-target, velocity, acceleration]
        return self.s

    def step(self, a):
        F = 10 * (a+1)
        position = self.s[0]
        velocity = self.s[1]
        acceleration = self.s[2]

        f = -self.f * np.sign(velocity) * velocity**2
        F = F + f - self.M*self.G

        acceleration_ = F/self.M
        velocity_ = velocity + self.t*(acceleration + acceleration_)/2
        position_ = position + self.t*(velocity + velocity_)/2

        self.s = [position_, velocity_, acceleration_]
        # 这种增量式的汇报函数设计，有正有负，可以更好地让智能体学习
        self.r = np.abs(position) - np.abs(position_)
        self.done = False
        return self.s, self.r, self.done, None


path = os.path.dirname(os.path.realpath(__file__))
env = GYM()
agent = PPO(path)

# Train Part
total_steps = 0
for episode in range(100):
    s = env.reset()
    ep_r = 0
    for ep_step in range(256):
        # interact environment
        a = agent.choose_action(s)
        s_, r, done, info = env.step(a[0])
        if ep_step >= 255:
            done = True
        # store transition into memory
        agent.store_transition(s, a, s_, r, done)
        s = s_
        ep_r += r
        total_steps += 1
    # show record
    print('Episode:', episode, ' Reward: %i' % int(ep_r))
    agent.store_network()

# Test Part
position = []
velocity = []
acceleration = []
action = []
s = env.reset()
action.append(0)
position.append(s[0])
velocity.append(s[1])
acceleration.append(s[2])
for ep_step in range(256):
    # interact environment
    a = agent.choose_action(s)

    s_, r, done, info = env.step(a[0])
    action.append(a[0])
    position.append(s[0])
    velocity.append(s[1])
    acceleration.append(s[2])

    s = s_

index = np.array(range(len(position)))
plt.plot(index, action, label='action')
plt.plot(index, position, label='position')
plt.plot(index, velocity, label='velocity')
plt.plot(index, acceleration, label='acc')
plt.legend()
plt.show()