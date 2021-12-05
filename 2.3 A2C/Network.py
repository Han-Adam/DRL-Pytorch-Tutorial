import torch.nn as nn


class Critic(nn.Module):
    # Critic Network, refer to the Network in DQN
    def __init__(self, s_dim, hidden):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(s_dim, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def forward(self, s):
        return self.critic(s)


class Actor(nn.Module):
    # Actor Network, refer to the Network in PolicyGradient
    def __init__(self, s_dim, hidden, a_num):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, a_num),
                                   nn.Softmax(dim=-1))

    def forward(self, s):
        return self.actor(s)