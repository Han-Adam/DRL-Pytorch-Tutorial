import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    def __init__(self, s_dim, hidden):
        super(VNet, self).__init__()

        self.linear1 = nn.Linear(s_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(QNet, self).__init__()

        self.linear1 = nn.Linear(s_dim + a_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(s_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)

        self.mean_linear = nn.Linear(hidden, a_dim)
        self.log_std_linear = nn.Linear(hidden, a_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return mean, std