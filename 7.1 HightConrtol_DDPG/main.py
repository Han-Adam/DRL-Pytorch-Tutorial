import gym
import numpy as np
import torch
from DDPG import DDPG
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
        # 被注释掉的是将目前状态与目标一起输入的情况，不稳定。
        # 用差分输入，高度控制就会好很多。
        # self.s = np.array([position, velocity, acceleration, target])
        self.s = np.array([position-target, velocity, acceleration])
        return self.s

    def step(self, a):
        F = 10 * (a+1)
        position = self.s[0]
        velocity = self.s[1]
        acceleration = self.s[2]
        # target = self.s[3]

        f = -self.f * np.sign(velocity) * velocity**2
        F = F + f - self.M*self.G

        acceleration_ = F/self.M
        velocity_ = velocity + self.t*(acceleration + acceleration_)/2
        position_ = position + self.t*(velocity + velocity_)/2

        self.s = np.array([position_, velocity_, acceleration_])
        # 回报函数的设置，设置增量或者绝对差值，都是可以的
        self.r = np.abs(position) - np.abs(position_)
        # self.s = np.array([position_, velocity_, acceleration_, target])
        # self.r = -np.abs(position_ - target)
        self.done = False
        return self.s, self.r, self.done, None


env = GYM()

# env = gym.make('Pendulum-v0')
# env = env.unwrapped

path = os.path.dirname(os.path.realpath(__file__))
agent = DDPG(path, s_dim=3)

# Train Part
total_steps = 0
for episode in range(50):
    s = env.reset()
    ep_r = 0
    for ep_step in range(200):
        # interact environment
        a = agent.choose_action(s)
        s_, r, done, info = env.step(a[0])
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
for ep_step in range(200):
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
