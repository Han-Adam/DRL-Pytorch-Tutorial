import gym
from Navigation2D import Navigation2D
import numpy as np
import torch
from Agent import DIAYN
import matplotlib.pyplot as plt


ITER = 40
env = Navigation2D(_iter=ITER)
S_DIM = env.observation_space.shape[0]          # state dimension
A_NUM = env.action_space.n                      # action dimension
SKILL_NUM = 4                                   # number of skills
HIDDEN = 256                                    # unit of hidden layer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAPACITY = 500                                  # maximum number of samples stored in memory
BATCH_SIZE = 256                                # the number of samples for each train
LR = 0.001                                       # learning rate
GAMMA = 0.9                                     # discounting factor
TAU = 0.05                                      # soft update parameter
LOG_PROB_REG = 1e-6                             # the minimum value of log_prob, to avoid log(0)
ALPHA = 0.1                                    # entropy regulation parameters
MAX_EPISODE = 50                                # maximum episode to play
START_LEARNING = 256                            # time to start to learn
RENDER = False                                  # whether render during training process

agent = DIAYN(s_dim=S_DIM,
              a_num=A_NUM,
              skill_num=SKILL_NUM,
              hidden=HIDDEN,
              lr=LR,
              gamma=GAMMA,
              tau=TAU,
              log_prob_reg=LOG_PROB_REG,
              alpha=ALPHA,
              capacity=CAPACITY,
              batch_size=BATCH_SIZE,
              device=DEVICE
            )

# training process
total_steps = 0
zs = np.eye(SKILL_NUM)
for episode in range(MAX_EPISODE):
    s = env.reset()
    z = zs[np.random.randint(0, SKILL_NUM)]
    ep_r = 0
    done = False
    while not done:
        if RENDER:
            env.render()
        # interact environment
        a = agent.get_action(s, z)
        s_, _, done, _ = env.step(a)
        r = agent.get_pseudo_reward(s, z, a, s_)
        # store in memory
        agent.memory.store_transition(s, z, a, s_, r, done)
        if total_steps > START_LEARNING:
            # time to learn
            agent.learn()
        # update record
        s = s_
        total_steps += 1
        ep_r += r
    print(episode)

# test process
trace = []
for i in range(SKILL_NUM):
    s = env.reset()
    z = zs[i]
    done = False
    sub_trace = []
    sub_trace.append(s)
    while not done:
        # env.render()
        a = agent.get_action(s, z)
        s_, _, done, _ = env.step(a)
        sub_trace.append(s_)
        s = s_
    trace.append(sub_trace)

for i in range(SKILL_NUM):
    sub_trace = trace[i]
    x = [sub_trace[j][0] for j in range(ITER)]
    y = [sub_trace[j][1] for j in range(ITER)]
    plt.plot(x, y, label='trace '+str(i))
plt.legend()

plt.show()
