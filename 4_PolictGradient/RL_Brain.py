import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
# reproducible
np.random.seed(1)


class RL_Brain:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95
    ):
        self.n_actions = n_actions
        # 这里的memory要直接存储一个 trajectory 的部分
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

        self.net = Net(n_input = n_features,
                       n_middle = 5*n_actions,
                       n_out = n_actions)
        self.optimiaze = Adam(self.net.parameters(), lr = learning_rate)
        self.Loss = Loss(self.net, n_actions, reward_decay)

    def choose_action(self, observation):
        prob_weights = self.net(torch.tensor(observation,dtype=torch.float))
        # select action w.r.t the actions prob
        action = (Categorical(prob_weights).sample()).detach().numpy()
        return action

    def store_transition(self, s, a, r):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)

    def learn(self):
        # train on episode
        loss = self.Loss(self.memory_s,self.memory_a,self.memory_r)
        self.optimiaze.zero_grad()
        loss.backward()
        self.optimiaze.step()
        # empty episode data
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []


class Net(nn.Module):
    def __init__(self,n_input, n_middle, n_out):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_input, n_middle, bias= True)
        self.out = nn.Linear(n_middle, n_out, bias = True)

    def forward(self, input):
        hidden = F.relu(self.hidden(input))
        output = F.softmax(self.out(hidden))
        return output


class Loss(nn.Module):
    def __init__(self, net, n_action, gamma):
        super(Loss,self).__init__()
        self.net = net
        self.n_action = n_action
        self.gamma = gamma

    def forward(self, memory_s, memory_a, memory_r):
        length = len(memory_r)
        print(length)

        memory_a = torch.tensor(np.array(memory_a), dtype= torch.long)
        discounted_reward = self._discount_and_norm_rewards(memory_r)

        probs = self.net(torch.tensor(memory_s, dtype=torch.float))
        m = Categorical(probs)
        print(m.probs)
        loss = -torch.sum(m.log_prob(memory_a)*torch.tensor(discounted_reward, dtype=torch.float))
        return loss

    def _discount_and_norm_rewards(self, memory_r):
        # discount episode rewards
        length = len(memory_r)
        discounted_r = np.zeros(length)
        running_add = 0
        for t in range(length-1, -1, -1):
            running_add = running_add * self.gamma + memory_r[t]
            discounted_r[t] = running_add
        # normalize episode rewards
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r