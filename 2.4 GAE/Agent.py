import copy

import torch
from torch.distributions import Categorical
from Network import Actor, Critic
import torch.nn.functional as F


class ActorCritic:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            hidden,
            lr_actor,
            lr_critic,
            memory_len,
            gamma,
            lambda_,
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.hidden = hidden
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.memory_len = memory_len
        self.gamma = gamma
        self.lambda_ = lambda_

        # network initialization
        self.actor = Actor(s_dim, hidden, a_num).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(s_dim, hidden).to(self.device)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # no memory in this algorithm
        self.memory_s = []
        self.memory_a = []
        self.memory_s_ = []
        self.memory_r = []
        self.memory_done = []

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
        self.memory_s_.append(s_)
        self.memory_r.append(r)
        self.memory_done.append(1 if done else 0)
        if len(self.memory_r)>self.memory_len:
            self._learn()

    def _GAE(self, s, r, s_, done):
        with torch.no_grad():
            v = self.critic(s).squeeze()
            v_ = self.critic(s_).squeeze()
            delta = r + self.gamma*v_*(1-done) - v

            length = r.shape[0]
            GAE = torch.zeros(size=[length], device=self.device)
            running_add = 0
            for t in range(length - 1, -1, -1):
                running_add = delta[t] + running_add * \
                              self.gamma * self.lambda_ * (1 - done[t])
                GAE[t] = running_add
            return GAE

    def _discounted_r(self, r, done):
        length = r.shape[0]
        discounted_r = torch.zeros([length], device=self.device)
        running_add = 0
        for t in range(length - 1, -1, -1):
            running_add = running_add * self.gamma * (1 - done[t]) + r[t]
            discounted_r[t] = running_add
        return discounted_r


    def _learn(self):
        # torch.LongTensor torch.FloatTensor only work for list
        # when transform scalar to Tensor, we could use torch.tensor()

        s = torch.tensor(self.memory_s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.memory_a, dtype=torch.long).to(self.device)
        s_ = torch.tensor(self.memory_s_, dtype=torch.float).to(self.device)
        r = torch.tensor(self.memory_r, dtype=torch.float).to(self.device)
        done = torch.tensor(self.memory_done, dtype=torch.float).to(self.device)
        GAE = self._GAE(s, r, s_, done)
        discounted_r = self._discounted_r(r, done)

        # update for critic
        v = self.critic(s).squeeze()
        critic_loss = F.mse_loss(v, discounted_r)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        # update for actor
        prob = self.actor(s)
        dist = Categorical(prob)
        actor_loss = -torch.sum(GAE.detach()*dist.log_prob(a))
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        self.memory_s = []
        self.memory_a = []
        self.memory_s_ = []
        self.memory_r = []
        self.memory_done = []