import torch
import torch.nn as nn


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain('tanh'))


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=32):
        super(Actor, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden),
                                     nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(hidden, a_dim),
                                nn.Tanh())
        # 这一波使用了可训练的，固定的std，也是可以用的
        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)
        # self.sigma = nn.Sequential(nn.Linear(hidden, a_dim),
        #                            nn.Softplus())
        init_linear(self)

    def forward(self, s):
        feature = self.feature(s)
        mu = self.mu(feature)
        sigma = self.log_std.exp()
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, s_dim, hidden=32):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(s_dim, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, 1))
        init_linear(self)

    def forward(self, s):
        return self.critic(s)
