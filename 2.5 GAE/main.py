import gym
import torch
from Agent import ActorCritic

env = gym.make('CartPole-v0')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_NUM = env.action_space.n                      # number of action
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR_ACTOR = 0.01                                 # learning rate for actor
LR_CRITIC = 0.01                                # learning rate for critic
GAMMA = 0.99                                     # discounting factor
LAMBDA = 0.95                                   # generalized factor
MAX_EPISODE = 500                               # maximum episode to play
MAX_STEP = 1e5                                  # maximum steps to play
MEMORY_LEN = 64                                 # the maximum length of memory
RENDER = False                                  # whether render

agent = ActorCritic(s_dim=S_DIM,
                    a_num=A_NUM,
                    device=DEVICE,
                    lr_actor=LR_ACTOR,
                    lr_critic=LR_CRITIC,
                    gamma=GAMMA,
                    lambda_=LAMBDA
                    )
total_steps = 0
episode = 0

while total_steps <= MAX_STEP:
    episode += 1
    s = env.reset()
    ep_r = 0
    while True:
        if RENDER:
            env.render()
        total_steps += 1
        # interact environment
        a = agent.get_action(s)
        s_, r, done, info = env.step(a)
        # reward reshaping
        r = -20 if done else r
        # learn
        agent.store_transition(s, a, s_, r, done)
        if total_steps%MEMORY_LEN == 0:
            agent.learn()
        # update record
        ep_r += r
        s = s_
        if done:
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2))
            break