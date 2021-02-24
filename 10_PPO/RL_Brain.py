import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RL_Brain:
    def __init__(self,
                 s_dim=3,
                 a_dim=1,
                 bound=2,
                 actor_lr=1e-4,
                 critic_lr=2e-4,
                 ppo_epoch=10,
                 gamma=0.9,
                 epsilon=0.2):
        # Parameter initialization
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.bound = bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.ppo_epoch = ppo_epoch
        self.gamma = gamma
        self.epsilon = epsilon
        # Actor Network
        self.actor = Actor(s_dim=self.s_dim,
                           a_dim=self.a_dim,
                           bound=self.bound)
        self.actor_old = Actor(s_dim=self.s_dim,
                               a_dim=self.a_dim,
                               bound=self.bound)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # Critic Network
        self.critic = Critic(s_dim=self.s_dim)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def select_action(self,s):
        s = torch.FloatTensor(s)
        mu, sigma = self.actor(s)
        dist = Normal(loc=mu, scale=sigma)
        a = dist.sample()
        a = torch.clamp(a, -self.bound, self.bound)
        # a.item(): convert one element Tensor to scalar
        return a.item()

    def get_v(self, s):
        s = torch.FloatTensor(s)
        with torch.no_grad():
            v = self.critic(s)
        return v.item()

    def calculate_log_prob(self, s, a, old=False):
        # s.dim = [batch, 3], a.dim = [batch, 1]
        # mu.dim = sigma.dim = log_prob.dim = [batch, 1]
        if old:
            with torch.no_grad():
                mu, sigma = self.actor_old(s)
        else:
            mu, sigma = self.actor(s)
        dist = Normal(loc=mu, scale=sigma)
        log_prob = dist.log_prob(a)
        return log_prob

    def update(self, s, a, r):
        # s.shape = [batch, 3]
        # a.shape = [batch, 1] => v.shape = [batch,1]
        # r.shape = [batch, 1] => advantage.shape = [batch, 1]
        self.actor_old.load_state_dict(self.actor.state_dict())
        old_log_prob = self.calculate_log_prob(s,a,old=True)
        with torch.no_grad():
            advantage = r - self.critic(s)
        for _ in range(self.ppo_epoch):
            self.update_actor(s, a, advantage, old_log_prob)
        for _ in range(self.ppo_epoch):
            self.update_critic(s, r)

    def update_actor(self, s, a, advantage, old_log_prob):
        # calculate actor loss (Clip version)
        log_prob = self.calculate_log_prob(s, a)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio*advantage
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon,
                                    1.0 + self.epsilon) * advantage
        loss = -torch.mean(torch.min(surr1,surr2))
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

class Actor(nn.Module):
    # input: state s.shape = [3] or [batch, 3]
    # output: mu, sigma shape = [1] or [batch, 1]
    def __init__(self, s_dim, a_dim, bound, hidden=100):
        super(Actor, self).__init__()
        self.input = nn.Linear(s_dim, hidden)
        self.mu = nn.Linear(hidden,a_dim)
        self.sigma = nn.Linear(hidden,a_dim)
        self.bound = bound

    def forward(self, s):
        input = torch.relu(self.input(s))
        mu = self.bound * torch.tanh(self.mu(input))
        sigma = F.softplus(self.sigma(input))
        return mu, sigma

class Critic(nn.Module):
    # input: state s.shape = [3] or [batch, 3]
    # output: value v.shape = [1] or [batch, 1]
    def __init__(self, s_dim, hidden=100):
        super(Critic, self).__init__()
        self.input = nn.Linear(s_dim, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, s):
        input = torch.relu(self.input(s))
        value = self.output(input)
        return value