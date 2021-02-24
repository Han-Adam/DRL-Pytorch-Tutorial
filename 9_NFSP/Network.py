import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class C_ReLU(nn.Module):
    # a special activation function
    def __init__(self):
        super(C_ReLU, self).__init__()

    def forward(self, x):
        # [B, C, L, H] --> [B, C*2, L, H]
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class conv_net(nn.Module):
    # [B, C, H, L] --> [B, 8*H*L]
    # this network is for the feature extraction
    # for both Q-net and Pi-net
    def __init__(self, s_dim):
        super(conv_net, self).__init__()
        self.s_dim = s_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=s_dim[0], out_channels=4, kernel_size=3, padding=1),
            C_ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            C_ReLU()
        )

    def forward(self, x):
        # view: [B, C, H, W] --> [B, C*H*W]
        result = self.net(x)
        result = result.view(x.size(0), -1)
        return result


class Q_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        # a simple DQN [B, *s_dim] --> [B, a_dim]
        super(Q_Net, self).__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim

        self.conv_net = conv_net(s_dim)
        self.middle_dim = self.conv_net(torch.zeros(size=[1, *self.s_dim])).shape[-1]
        self.value_net = nn.Sequential(
            nn.Linear(self.middle_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.a_dim)
        )

    def forward(self, x):
        return self.value_net(self.conv_net(x))


class Pi_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        # a simple DQN [B, *s_dim] --> [B, a_dim]
        # additional Softmax layer is added
        super(Pi_Net, self).__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim

        self.conv_net = conv_net(s_dim)
        self.middle_dim = self.conv_net(torch.zeros(size=[1, *self.s_dim])).shape[-1]
        self.value_net = nn.Sequential(
            nn.Linear(self.middle_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.a_dim)
        )

    def forward(self, x):
        return F.softmax(self.value_net(self.conv_net(x)), dim=-1)
