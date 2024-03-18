import gym
import torch
from Agent import ActorCritic

env = gym.make('CartPole-v1')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_NUM = env.action_space.n                      # number of action
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN = 32                                     # hidden node for network
LR_ACTOR = 0.01                                 # learning rate for actor
LR_CRITIC = 0.01                                # learning rate for critic
GAMMA = 0.9                                     # discounting factor
MAX_EPISODE = 300                               # maximum episode to play
RENDER = False                                  # whether render

agent = ActorCritic(s_dim=S_DIM,
                    a_num=A_NUM,
                    device=DEVICE,
                    hidden=HIDDEN,
                    lr_actor=LR_ACTOR,
                    lr_critic=LR_CRITIC,
                    gamma=GAMMA
                    )
total_steps = 0

for episode in range(MAX_EPISODE):
    s, _ = env.reset()
    ep_r = 0
    while True:
        if RENDER:
            env.render()
        # interact environment
        a = agent.get_action(s)
        s_, r, done, _, _ = env.step(a)
        # reward reshaping
        r = -20 if done else r
        # learn
        agent.learn(s, a, s_, r, done)
        # update record
        ep_r += r
        s = s_
        total_steps += 1
        if done:
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2))
            break
