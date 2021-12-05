import torch.nn as nn


class Q_Net(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(Q_Net, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden),
                                     nn.ReLU())
        self.v = nn.Sequential(nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden),
                                       nn.ReLU(),
                                       nn.Linear(hidden, a_num))

    def forward(self, s):
        feature = self.feature(s)
        v = self.v(feature)
        advantage = self.advantage(feature)
        q = v + advantage
        return q
