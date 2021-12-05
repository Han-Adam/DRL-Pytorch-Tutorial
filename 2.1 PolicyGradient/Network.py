import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, hidden),
                                 nn.ReLU(),
                                 nn.Linear(hidden, a_num),
                                 nn.Softmax())

    def forward(self, s):
        return self.net(s)
