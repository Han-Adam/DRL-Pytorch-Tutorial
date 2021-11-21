import torch
import torch.nn as nn


def init_linear(module, std=1, bias=1e-6):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, std)
            nn.init.constant_(m.bias, bias)


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Actor, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(hidden, a_dim),
                                nn.Tanh())
        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)
        init_linear(self)

    def forward(self, s):
        feature = self.feature(s)
        mu = self.mu(feature)
        sigma = self.log_std.exp()
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, s_dim, hidden):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(s_dim, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, 1))
        init_linear(self)

    def forward(self, s):
        return self.critic(s)
