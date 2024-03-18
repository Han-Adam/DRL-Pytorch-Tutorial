import gym
import numpy as np
import torch
from Agent import TD3

env = gym.make('Pendulum-v1')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_DIM = env.action_space.shape[0]               # action dimension
BOUND = env.action_space.high[0]                # bound value of the action
CAPACITY = 10000                                # maximum number of samples stored in memory
BATCH_SIZE = 256                                # the number of samples for each train
HIDDEN = 32                                     # hidden node
LR_ACTOR = 5e-3                                 # learning rate for actor
LR_CRITIC = 5e-3                                # learning rate for critic
VARIANCE_START = 1                              # the initial random factor
VARIANCE_DECAY = 0.999                          # the decay rate of random factor for each step
VARIANCE_MIN = 0.05                             # the minimum random factor
GAMMA = 0.9                                     # discounting factor
TAU = 0.05                                      # soft-update parameters
POLICY_NOISE = 0.2                              # the sigma for policy noise
NOISE_CLIP = 0.5                                # noise clip
POLICY_FREQ = 2                                 # frequency of update policy
MAX_EPISODE = 1000                              # maximum episode to play
MAX_EP_STEPS = 256                              # maximum steps for each episode
RENDER = False                                  # whether render


agent = TD3(s_dim=S_DIM,
            a_dim=A_DIM,
            capacity=CAPACITY,
            batch_size=BATCH_SIZE,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            hidden=HIDDEN,
            var_init=VARIANCE_START,
            var_decay=VARIANCE_DECAY,
            var_min=VARIANCE_MIN,
            gamma=GAMMA,
            tau=TAU,
            policy_noise=POLICY_NOISE,
            noise_clip=NOISE_CLIP,
            policy_freq=POLICY_FREQ)

# randomly get some samples
for episode in range(4):
    s, _ = env.reset()
    ep_r = 0
    for ep_step in range(MAX_EP_STEPS):
        a = env.action_space.sample()
        s_, r, done, _, _ = env.step(a * BOUND)
        agent.memory.store_transition(s, a, s_, r/10, done)
        s = s_

for episode in range(30):
    s, _ = env.reset()
    ep_r = 0
    for ep_step in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        a = agent.get_action(s)
        s_, r, done, _, _ = env.step(a*BOUND)
        agent.memory.store_transition(s, a, s_, r/10, done)
        agent.learn()
        s = s_
        ep_r += r
    # 实际上，我们把learn放到循环里面还是外面，收敛区别不大
    # 根据Revisiting Fundamentals of Experience Replay
    # 重要的是replay ratio: the number of gradient updates per environment transition
    # for i in range(MAX_EP_STEPS):
    #     agent.learn()
    print('Episode:', episode, ' Reward: %i' % int(ep_r), 'var', agent.var)
    # if ep_r > -300:
    #     RENDER = True