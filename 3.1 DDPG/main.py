import gym
import torch
from Agent import DDPG

env = gym.make('Pendulum-v1')
env = env.unwrapped

S_DIM = env.observation_space.shape[0]          # state dimension
A_DIM = env.action_space.shape[0]               # action dimension
print(S_DIM, A_DIM)
BOUND = env.action_space.high[0]                # bound value of the action
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN = 32
CAPACITY = 10000                                # maximum number of samples stored in memory
BATCH_SIZE = 128                                # the number of samples for each train
LR_ACTOR = 0.005                                # learning rate for actor
LR_CRITIC = 0.005                               # learning rate for critic
VARIANCE_START = 1                              # the initial random factor
VARIANCE_DECAY = 0.9999                         # the decay rate of random factor for each step
VARIANCE_MIN = 0.05                             # the minimum random factor
GAMMA = 0.9                                     # discounting factor
TAU = 0.05                                      # soft-update parameters
MAX_EPISODE = 1000                              # maximum episode to play
MAX_EP_STEPS = 256                              # maximum steps for each episode
START_LEARN_STEP = 256                          # the step for agent start to learn
RENDER = False                                  # whether render


agent = DDPG(s_dim=S_DIM,
             a_dim=A_DIM,
             device=DEVICE,
             hidden=HIDDEN,
             capacity=CAPACITY,
             batch_size=BATCH_SIZE,
             lr_actor=LR_ACTOR,
             lr_critic=LR_CRITIC,
             variance_start=VARIANCE_START,
             variance_decay=VARIANCE_DECAY,
             variance_min=VARIANCE_MIN,
             gamma=GAMMA,
             tau=TAU)

total_steps = 0
for episode in range(MAX_EPISODE):
    s, _ = env.reset()
    ep_r = 0
    for ep_step in range(MAX_EP_STEPS):
        if RENDER: env.render()
        # interact environment
        a = agent.get_action(s)
        s_, r, done, _, _ = env.step(a*BOUND)
        # store transition into memory
        agent.memory.store_transition(s, a, s_, r/10, done)
        # time to learn
        if total_steps > START_LEARN_STEP:
            agent.learn()
        # renew recoder
        s = s_
        ep_r += r
        total_steps += 1
    # show record
    print('Episode:', episode, ' Reward: %i' % int(ep_r))
    # if ep_r > -300:
    #     RENDER = True