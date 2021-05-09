import torch.nn as nn


class Critic(nn.Module):
    # Critic Network, refer to the Network in DQN
    def __init__(self, n_input, n_hidden):
        super(Critic,self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, s):
        return self.critic(s)


class Actor(nn.Module):
    # Actor Network, refer to the Network in PolicyGradient
    def __init__(self, s_dim, hidden, a_num):
        super(Actor,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_num),
            nn.Softmax()
        )

    def forward(self, s):
        return self.net(s)