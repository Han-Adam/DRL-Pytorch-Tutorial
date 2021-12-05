import gym
import torch
from Agent import PolicyGradient

env = gym.make('CartPole-v0')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_NUM = env.action_space.n                      # number of action
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN = 32                                     # hidden node for network
LR = 0.01                                       # learning rate
GAMMA = 0.9                                     # discounting factor
MAX_EPISODE = 300                               # maximum episode to play
RENDER = False                                  # whether render

agent = PolicyGradient(s_dim=S_DIM,
                       a_num=A_NUM,
                       device=DEVICE,
                       hidden=HIDDEN,
                       lr=LR,
                       gamma=GAMMA
                       )
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
        # store transition into memory
        agent.store_transition(s, a, r)
        # update record
        ep_r += r
        s = s_
        total_steps += 1
        if done:
            # PolicyGradient only learns at the end of each game
            agent.learn()
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2))
            break
