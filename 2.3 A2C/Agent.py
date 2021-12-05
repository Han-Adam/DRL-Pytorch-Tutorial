import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from Network import Actor, Critic


class A2C:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            hidden,
            lr_actor,
            lr_critic,
            max_len,
            gamma,
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.hidden = hidden
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.max_len = max_len
        self.gamma = gamma

        # network initialization
        self.actor = Actor(s_dim, hidden, a_num).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(s_dim, hidden).to(self.device)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # define memory
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

    def get_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        prob_weights = self.actor(s)
        # select action w.r.t the actions prob
        dist = Categorical(prob_weights)
        action = (dist.sample()).detach().item()
        return action

    def store_transition(self, s, a, s_, r, done):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)
        if len(self.memory_r) >= self.max_len or done:
            discounted_r = self._discounted_r(self.memory_r, s_, done)
            s = torch.FloatTensor(self.memory_s).to(self.device)
            a = torch.LongTensor(self.memory_a).to(self.device)
            r = torch.FloatTensor(discounted_r).to(self.device)
            self._learn(s, a, r)

    def _learn(self, s, a, r):
        # update critic
        v = self.critic(s)
        advantage = r - v
        critic_loss = torch.mean(torch.pow(advantage, 2))
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        # update actor
        prob = self.actor(s)
        dist = Categorical(prob)
        log_prob = dist.log_prob(a)
        actor_loss = -torch.mean(log_prob * advantage.detach())
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()
        # renew the memory
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

    def _discounted_r(self, r, s_, done):
        length = len(r)
        discounted_r = np.zeros(length)
        running_add = 0 if done else self.critic(torch.FloatTensor(s_).to(self.device)).item()
        for t in range(length - 1, -1, -1):
            running_add = r[t] + running_add * self.gamma
            discounted_r[t] = running_add
        return discounted_r
