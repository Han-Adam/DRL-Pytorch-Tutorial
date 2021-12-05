import gym
import numpy as np
import torch
from Agent import SAC

# Soft Actor-Critic Algorithms and Applications
# https://arxiv.org/pdf/1812.05905.pdf
env = gym.make('Pendulum-v0')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_DIM = env.action_space.shape[0]               # action dimension
HIDDEN = 256                                    # node size for hidden layer
BOUND = env.action_space.high[0]                # bound value of the action
CAPACITY = 10000                                # maximum number of samples stored in memory
BATCH_SIZE = 128                                # the number of samples for each train
LR = 0.001                                      # learning rate
GAMMA = 0.9                                     # discounting factor
TAU = 0.05                                      # soft update parameter
LOG_PROB_REG = 1e-6                             # the minimum value of log_prob, to avoid log(0)
MAX_EPISODE = 1000                              # maximum episode to play
MAX_EP_STEPS = 256                              # maximum steps for each episode
START_LEARNING = 1000                           # time to start to learn
RENDER = False                                  # whether render


agent = SAC(s_dim=S_DIM,
            a_dim=A_DIM,
            hidden=HIDDEN,
            capacity=CAPACITY,
            batch_size=BATCH_SIZE,
            lr=LR,
            gamma=GAMMA,
            tau=TAU,
            log_prob_reg=LOG_PROB_REG
            )
total_steps = 0

for episode in range(MAX_EPISODE):
    s = env.reset()
    ep_r = 0
    for ep_step in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        a = agent.get_action(s)
        s_, r, done, info = env.step(BOUND*a[0])
        agent.memory.store_transition(s, a, s_, r)
        s = s_
        total_steps += 1
        ep_r += r
    if total_steps > START_LEARNING:
        for i in range(MAX_EP_STEPS):
            agent.learn()
    print('episode: ', episode,
          ' ep_r: ', round(ep_r, 2),
          ' alpha: ', agent.alpha)
