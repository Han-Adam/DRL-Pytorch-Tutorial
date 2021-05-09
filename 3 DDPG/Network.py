import torch
import torch.nn as nn


class Critic(nn.Module):
    # Critic Network
    # input: the state and corresponding action
    # output: the q value
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic,self).__init__()
        self.input_s = nn.Linear(s_dim, hidden)
        self.input_a = nn.Linear(a_dim, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, s, a):
        input_s = self.input_s(s)
        input_a = self.input_a(a)
        input_total = torch.relu(torch.add(input_a, input_s))
        q = self.output(input_total)
        return q


class Actor(nn.Module):
    # Action Network
    # input: the state
    # output: the action
    def __init__(self, s_dim, hidden, a_dim, bound):
        super(Actor,self).__init__()
        self.bound = torch.tensor(bound, dtype=torch.float)
        self.actor = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim),
            nn.Tanh(),
        )

    def forward(self, s):
        return self.actor(s)*self.bound