import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.critic = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.critic(s)

