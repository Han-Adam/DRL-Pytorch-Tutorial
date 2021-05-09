'''
The main frame is from OpenAI's PPO <Emergence of Locomotion Behaviours in Rich Environments>
But the clip loss function is from DeepMind's <Proximal Policy Optimization Algorithms>
'''
import gym
from Agent import PPO
import torch

env = gym.make('Pendulum-v0')
env = env.unwrapped
S_DIM = env.observation_space.shape[0]          # state dimension
A_DIM = env.action_space.shape[0]               # action dimension
BOUND = env.action_space.high[0]                # bound value of the action
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR_ACTOR = 0.001                               # learning rate for actor
LR_CRITIC = 0.002                              # learning rate for critic
MAX_LEN = 32                                    # the max length trajectory stored in memory
GAMMA = 0.9                                     # discounting factor
EPSILON = 0.2                                   # threshold for the probability ratio
TAU = 0.05                                      # soft-update parameters
UPDATE_STEPS_A = 10                             # update step for actor
UPDATE_STEPS_C = 10                             # update step for critic
MAX_EPISODE = 1000                              # maximum episode to play
MAX_EP_STEPS = 200                              # maximum steps for each episode
RENDER = False                                  # whether render

agent = PPO(s_dim=S_DIM,
            a_dim=A_DIM,
            bound=BOUND,
            actor_lr=LR_ACTOR,
            critic_lr=LR_CRITIC,
            update_step_a=UPDATE_STEPS_A,
            update_step_c=UPDATE_STEPS_C,
            gamma=GAMMA,
            epsilon=EPSILON)


for episode in range(MAX_EPISODE):
    s = env.reset()
    ep_r = 0
    for t in range(MAX_EP_STEPS):
        # interact with environment
        a = agent.get_action(s)
        s_, r, done, _ = env.step(a)
        if (t+1) % MAX_LEN == 0 or t == MAX_EP_STEPS-1:
            done = True
        # learn
        agent.learn(s, a, s_, (r+8)/8, done)
        # update state
        s = s_
        ep_r += r
    print('Episode:', episode, ' Reward: %i' % int(ep_r))