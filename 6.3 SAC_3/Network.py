import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.mean = nn.Sequential(nn.Linear(s_dim, hidden),
                                  nn.ReLU(),
                                  nn.Linear(hidden, hidden),
                                  nn.ReLU(),
                                  nn.Linear(hidden, a_dim))

        self.log_std = nn.Parameter(torch.ones([a_dim]), requires_grad=True)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s):
        mean = self.mean(s)
        std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max).exp()
        std = torch.ones_like(mean) * std
        return mean, std


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden),
                                nn.ReLU(),
                                nn.Linear(hidden, hidden),
                                nn.ReLU(),
                                nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden),
                                nn.ReLU(),
                                nn.Linear(hidden, hidden),
                                nn.ReLU(),
                                nn.Linear(hidden, 1))

    def forward(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q1 = self.q1(s_a)
        q2 = self.q2(s_a)
        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q1 = self.q1(s_a)
        return q1
