import gym
import numpy as np
import torch
from Agent import DQN

env = gym.make('CartPole-v0')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_NUM = env.action_space.n                      # number of action
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAPACITY = 500                                  # maximum number of samples stored in memory
BATCH_SIZE = 32                                 # the number of samples for each train
LR = 0.01                                       # learning rate
EPSILON_START = 0.1                             # the initial random factor
GREEDY_INCREASE = 0.001                         # the increment of random factor for each step
GAMMA = 0.9                                     # discounting factor
REPLACE_TARGET_ITER = 300                       # steps for apply hard replace
MAX_EPISODE = 300                               # maximum episode to play
START_LEARNING = 100                            # time to start to learn
RENDER = True                                   # whether render

agent = DQN(s_dim=S_DIM,
            a_num=A_NUM,
            device=DEVICE,
            capacity=CAPACITY,
            batch_size=BATCH_SIZE,
            lr=LR,
            epsilon_start=EPSILON_START,
            greedy_increase=GREEDY_INCREASE,
            gamma=GAMMA,
            replace_target_iter=REPLACE_TARGET_ITER)
total_steps = 0

for episode in range(MAX_EPISODE):
    s = env.reset()
    ep_r = 0
    while True:
        if RENDER:
            env.render()
        # interact environment
        a = agent.get_action(s)
        s_, r, done, info = env.step(a)
        # reward reshaping
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        r = r1 + r2
        # store transition into memory
        agent.memory.store_transition(s, a, s_, r, done)
        ep_r += r
        if total_steps > START_LEARNING:
            # time to learn
            agent.learn()

        s = s_
        total_steps += 1
        if done:
            # show some result and enter into next episode
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(agent.epsilon, 2))
            break