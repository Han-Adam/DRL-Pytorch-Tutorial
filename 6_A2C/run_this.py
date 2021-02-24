import numpy as np
from RL_Brain import RL_Brain
import gym
import torch

np.random.seed(2)

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic
LR_STEP = 5     # the steps to learn

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

RL = RL_Brain(
            n_actions=N_A,
            n_features=N_F,
            lr_actor=0.01,
            lr_critic=0.001,
            reward_decay=0.9,
)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []

    m_s = []
    m_a = []
    m_r = []

    while True:
        if RENDER: env.render()

        a = RL.choose_action(s)

        s_, r, done, info = env.step(a.numpy())

        if done: r = -20

        track_r.append(r)

        m_s.append(s)
        m_a.append(a)
        m_r.append(r)
        if len(m_s)>= LR_STEP or done:
            # learn
            discounted_r = []
            value_ = 0 if done else RL.critic(torch.FloatTensor(s_)).item()
            for t in range(len(m_s)-1, -1, -1):
                value_ = value_ * GAMMA + m_r[t]
                discounted_r.insert(0, value_)
            RL.learn(m_s, m_a, discounted_r)

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)+20
            running_reward = ep_rs_sum
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break