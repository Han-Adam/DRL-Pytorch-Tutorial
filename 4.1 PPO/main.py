'''
This is a single agent demo of  DeepMind's <Proximal Policy Optimization Algorithms>
Compared with PPO1, this algorithm contains GAE, Batch Update.
PPO3相较于PPO2，加大了batch，需要考虑done的部分。
'''
import gym
from Agent import PPO
import os
import yaml

if __name__ == '__main__':
    # Read Parameters:
    param_path = os.path.dirname(__file__) + '/HyperParameters.yaml'
    with open(param_path, 'r', encoding='utf-8') as F:
        param_dict = yaml.load(F, Loader=yaml.FullLoader)
    MAX_EPISODE = param_dict['MAX_EPISODE']
    MAX_STEP = param_dict['MAX_STEP']
    S_DIM = param_dict['S_DIM']
    A_DIM = param_dict['A_DIM']
    BOUND = param_dict['BOUND']
    HIDDEN = param_dict['HIDDEN']
    DEVICE = param_dict['DEVICE']
    GAMMA = param_dict['GAMMA']
    LAMBDA = param_dict['LAMBDA']
    EPSILON = param_dict['EPSILON']
    LR = param_dict['LR']
    MEMORY_LEN = param_dict['MEMORY_LEN']
    BATCH_SIZE = param_dict['BATCH_SIZE']
    UPDATE_EPOCH = param_dict['UPDATE_EPOCH']

    RENDER = False

    # Construct environment and agent
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    agent = PPO(s_dim=S_DIM,
                a_dim=A_DIM,
                bound=BOUND,
                hidden=HIDDEN,
                device=DEVICE,
                lr=LR,
                memory_len=MEMORY_LEN,
                batch_size=BATCH_SIZE,
                update_epoch=UPDATE_EPOCH,
                gamma=GAMMA,
                lambda_=LAMBDA,
                epsilon=EPSILON)

    # Train process
    for episode in range(MAX_EPISODE):
        s = env.reset()
        ep_r = 0.
        for step in range(MAX_STEP):
            a = agent.get_action(s)
            s_, r, done, _ = env.step(a)
            if step == MAX_STEP-1:
                done = True
            agent.learn(s, a, s_, (r+8)/8, done)
            s = s_
            ep_r += r
        print('Episode:', episode, ' Reward: %i' % int(ep_r))