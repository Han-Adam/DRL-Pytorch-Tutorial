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
GAMMA = 0.99                                     # discounting factor
LAMBDA = 0.95                                   # generalized factor
MAX_EPISODE = 500                               # maximum episode to play
MAX_STEP = 1e5                                  # maximum steps to play
MEMORY_LEN = 64                                 # the maximum length of memory
RENDER = False                                  # whether render

agent = ActorCritic(s_dim=S_DIM,
                    a_num=A_NUM,
                    device=DEVICE,
                    hidden=HIDDEN,
                    lr_actor=LR_ACTOR,
                    lr_critic=LR_CRITIC,
                    memory_len=MEMORY_LEN,
                    gamma=GAMMA,
                    lambda_=LAMBDA
                    )
total_steps = 0
episode = 0

while total_steps <= MAX_STEP:
    episode += 1
    s, _ = env.reset()
    ep_r = 0
    while True:
        if RENDER:
            env.render()
        total_steps += 1
        a = agent.get_action(s)
        s_, r, done, _, _ = env.step(a)
        # reward reshaping
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        agent.store_transition(s, a, s_, r, done)
        ep_r += r
        s = s_
        if done:
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2))
            break