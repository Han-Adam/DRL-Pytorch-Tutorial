import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(s_dim+a_dim, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def forward(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q = self.critic(s_a)
        return q


class Actor(nn.Module):
    def __init__(self, s_dim, hidden, a_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, a_dim),
                                   nn.Tanh())

    def forward(self, s):
        return self.actor(s)
