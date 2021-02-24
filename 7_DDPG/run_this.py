import numpy as np
import gym
import time
from RL_Brain import RL_Brain
import matplotlib.pyplot as plt

np.random.seed(1)
#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.1
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

rl_brain = RL_Brain(
                state_dim = state_dim,
                action_dim = action_dim,
                action_bound = action_bound,
                memory_capacity = MEMORY_CAPACITY,
                tau = TAU,
                gamma = GAMMA,
                lr_actor = LR_A,
                lr_critic = LR_C,
                batch_size = BATCH_SIZE,
                )

var = 0  # control exploration
trace = []

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        # Add exploration noise
        a = rl_brain.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        rl_brain.store_transition(s, a, r / 10, s_)

        if rl_brain.memory.counter > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            rl_brain.learn()

        s = s_
        ep_reward += r


        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            trace.append(1+ep_reward/2000)
            # if ep_reward > -300:
            #     RENDER = True
            break

plt.plot(trace)
plt.show()

print('Running time: ', time.time()-t1)