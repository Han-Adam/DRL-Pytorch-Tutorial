import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from Network import Actor, Critic

class PPO:
    def __init__(self,
                 s_dim=3,
                 a_dim=1,
                 bound=2,
                 actor_lr=1e-4,
                 critic_lr=2e-4,
                 update_step_a=10,
                 update_step_c=10,
                 gamma=0.9,
                 epsilon=0.2):
        # Parameter initialization
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.bound = bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_step_a = update_step_a
        self.update_step_c = update_step_c
        self.gamma = gamma
        self.epsilon = epsilon

        # network initialization
        self.actor = Actor(s_dim, a_dim, bound)
        self.actor_old = Actor(s_dim, a_dim, bound)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic = Critic(s_dim)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # memory initialization
        self.memory_s, self.memory_a, self.memory_r = [], [], []

    def get_action(self, s):
        # select action w.r.t the actions prob
        s = torch.FloatTensor(s)
        mu, sigma = self.actor(s)
        dist = Normal(loc=mu, scale=sigma)
        a = dist.sample()
        a = torch.clamp(a, -self.bound, self.bound)
        return a.item()

    def get_v(self, s):
        # the state value
        s = torch.FloatTensor(s)
        with torch.no_grad():
            v = self.critic(s)
        return v.item()

    def calculate_log_prob(self, s, a, old=False):
        # s.shape = [batch, s_dim], a.shape = [batch, a_dim]
        # mu.shape = sigma.shape = log_prob.shape = [batch, a_dim]
        if old:
            with torch.no_grad():
                mu, sigma = self.actor_old(s)
        else:
            mu, sigma = self.actor(s)
        dist = Normal(loc=mu, scale=sigma)
        log_prob = dist.log_prob(a)
        return log_prob

    def learn(self, s, a, s_, r, done):
        # store transition
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)
        if done:
            # calculate the discounted reward
            discounted_r = []
            v_ = self.get_v(s_)
            for t in range(len(self.memory_r) - 1, -1, -1):
                v_ = self.memory_r[t] + self.gamma * v_
                discounted_r.insert(0, v_)
            s = torch.FloatTensor(self.memory_s)
            a = torch.FloatTensor(self.memory_a).unsqueeze(dim=-1)
            r = torch.FloatTensor(discounted_r).unsqueeze(dim=-1)
            # start to update network
            self.actor_old.load_state_dict(self.actor.state_dict())
            old_log_prob = self.calculate_log_prob(s, a, old=True)
            with torch.no_grad():
                advantage = r - self.critic(s)
            for _ in range(self.update_step_a):
                self.update_actor(s, a, advantage, old_log_prob)
            for _ in range(self.update_step_c):
                self.update_critic(s, r)
            # empty the memory
            self.memory_s, self.memory_a, self.memory_r = [], [], []

    def update_actor(self, s, a, advantage, old_log_prob):
        # calculate the loss
        log_prob = self.calculate_log_prob(s, a)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio*advantage
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon,
                                    1.0 + self.epsilon) * advantage
        loss = -torch.mean(torch.min(surr1, surr2))
        # update
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def update_critic(self, s, r):
        # calculate critic loss
        v = self.critic(s)
        advantage = r - v
        loss = torch.mean(advantage**2)
        # update
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()