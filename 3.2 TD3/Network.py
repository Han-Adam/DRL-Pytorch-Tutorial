import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, a_dim),
                                   nn.Tanh())

    def forward(self, s):
        return self.actor(s)


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden),
                                nn.ReLU(),
                                nn.Linear(hidden, 1))

        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden),
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
