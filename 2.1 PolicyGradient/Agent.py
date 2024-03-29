import numpy as np
import torch
from torch.distributions import Categorical
from Network import Net


class PolicyGradient:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            hidden,
            lr,
            gamma,
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.hidden = hidden
        self.lr = lr
        self.gamma = gamma

        # network initialization
        self.net = Net(s_dim, hidden, a_num).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        # the memory only need to store a trajectory
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

    def get_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        prob_weights = self.net(s)
        # select action w.r.t the actions prob
        dist = Categorical(prob_weights)
        action = (dist.sample()).detach().item()
        return action

    def store_transition(self, s, a, r):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)

    def learn(self):
        discounted_r = self._discounted_r(self.memory_r)
        s = torch.FloatTensor(self.memory_s).to(self.device)
        a = torch.LongTensor(self.memory_a).to(self.device)
        r = torch.FloatTensor(discounted_r).to(self.device)
        # calculate loss
        prob = self.net(s)
        dist = Categorical(prob)
        loss = -torch.sum(dist.log_prob(a) * r)
        # train on episode
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # empty episode data
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

    def _discounted_r(self, r):
        length = len(r)
        discounted_r = np.zeros(length)
        running_add = 0
        for t in range(length - 1, -1, -1):
            running_add = r[t] + running_add * self.gamma
            discounted_r[t] = running_add
        # normalize episode rewards
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r
