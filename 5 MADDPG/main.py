import numpy as np
import torch
from make_env import make_env
from agent import MADDPG

# parameter setting for environment
ENV_NAME = 'simple_push'#'simple_adversary'       # environment name
EPISODE_MAX_LEN = 45                # the max length for each episode
MAX_EPISODE = 1500                 # max episode number
ADVERSARIES_NUM = 1                 # the number of adversaries
# parameter setting for network training
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
START_LEARNING_STEP = 500           # the step that we statr to learn
MAX_GRAD_NORM = 0.5                 # clip the gradient
LEARNING_FREQUENCY = 100            # learning frequency
GAMMA = 0.97                        # discounting rate
TAU = 0.01                          # replace rate for soft-update
LR_A = 1e-2                         # learning rate for actor
LR_C = 1e-2                         # learning rate for critic
BATCH_SIZE = 256                    # batch size
BUFFER_CAPACITY = int(5e4)        # buffer capacity
HIDDEN = 64                        # nums of neural for hidden layer
STORE_PATH = './'

if __name__ == '__main__':
    # definition of environment
    env = make_env(ENV_NAME)
    state_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)] # [8, 10, 10]
    action_shape_n = [env.action_space[i].n for i in range(env.n)]            # [5, 5, 5]
    adversaries_num = min(env.n, ADVERSARIES_NUM)                             # 1
    # definition of agent
    agent = MADDPG(device=DEVICE,
                   agent_num=env.n,
                   state_shape_n=state_shape_n,
                   action_shape_n=action_shape_n,
                   gamma=GAMMA,
                   tau=TAU,
                   max_grad_norm=MAX_GRAD_NORM,
                   hidden=HIDDEN,
                   lr_a=LR_A,
                   lr_c=LR_C,
                   buffer_capacity=BUFFER_CAPACITY,
                   batch_size=BATCH_SIZE,
                   )
    # start to train
    s_n = env.reset()
    total_step = 0
    reward_history = []
    for episodes in range(MAX_EPISODE):
        episode_reward = [0]*env.n
        episode_step = 0
        while True:
            # get action from the agent
            a_n = agent.get_action(s_n)
            # interact with the environment
            s_n_, r_n, done_n, info = env.step(a_n)
            # store the transition
            agent.memory.store_transition(s_n, a_n, s_n_, r_n, done_n)
            # train the agent
            if total_step>START_LEARNING_STEP and \
                (total_step-START_LEARNING_STEP)%LEARNING_FREQUENCY == 0:
                agent.train()
            # update step
            total_step += 1
            episode_step += 1
            for i in range(env.n):
                episode_reward[i] += r_n[i]
            # if game over or max length is reached, start a new episode
            if all(done_n) or (episode_step >= EPISODE_MAX_LEN):
                s_n = env.reset()
                reward_history.append(episode_reward)
                print(episodes, total_step, episode_reward)
                break
            else:
                s_n = s_n_
    # store the trained model and history
    for i in range(env.n):
        torch.save(agent.actors[i].state_dict(), STORE_PATH+'agent'+str(i)+'.pt')
    np.save(STORE_PATH+'history.npy', np.array(reward_history))
