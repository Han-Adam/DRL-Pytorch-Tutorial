'''
The main frame is from OpenAI's PPO <Emergence of Locomotion Behaviours in Rich Environments>
But the clip loss function is from DeepMind's <Proximal Policy Optimization Algorithms>
'''
import gym
from RL_Brain import RL_Brain
import torch

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
EPSILON = 0.2
RENDER = False
env_name = 'Pendulum-v0'
env = gym.make(env_name)
brain = RL_Brain()


for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        #env.render()
        # a: scalar, need to convert to list
        # s: [] list of len()=3
        # r: scalar
        a = brain.select_action(s)
        s_, r, done, _ = env.step([a])
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            # calculate discounted reward
            v_s_ = brain.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            # update network
            bs, ba, br = torch.FloatTensor(buffer_s), \
                         torch.FloatTensor(buffer_a).unsqueeze(dim = -1), \
                         torch.FloatTensor(discounted_r).unsqueeze(dim = -1)
            buffer_s, buffer_a, buffer_r = [], [], []
            # bs.shape = [batch, 3]
            # ba.shape = br.shape = [batch, 1]
            brain.update(bs, ba, br)
    print(ep, int(ep_r))