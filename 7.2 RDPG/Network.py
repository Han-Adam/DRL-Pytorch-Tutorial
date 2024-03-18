import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, num_layers):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(s_dim + a_dim, hidden, batch_first=True, num_layers=num_layers)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def forward(self, s, a, h0=None, c0=None, return_hidden=False):
        hidden = None if h0 is None and c0 is None else (h0, c0)
        s_a = torch.cat([s, a], dim=-1)
        feature, (hn, cn) = self.lstm(s_a, hidden)
        q = self.critic(feature)
        if return_hidden:
            return q, hn, cn
        else:
            return q
        # s_a = torch.cat([s, a], dim=-1)
        # q = self.critic(s_a)
        # return q


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, num_layers):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(s_dim, hidden, batch_first=True, num_layers=num_layers)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, a_dim),
                                   nn.Tanh())

    def forward(self, s, h0=None, c0=None, return_hidden=False):
        hidden = None if h0 is None and c0 is None else (h0, c0)
        feature, (hn, cn) = self.lstm(s, hidden)
        a = self.actor(feature)
        if return_hidden:
            return a, hn, cn
        else:
            return a
        # return self.actor(s)

# class DRQ_Net(nn.Module):
#     def __init__(self, s_dim, hidden, a_num, num_layers=1):
#         super(DRQ_Net, self).__init__()
#         self.lstm = nn.LSTM(s_dim, hidden, batch_first=True, num_layers=num_layers)
#         self.net = nn.Sequential(nn.Linear(hidden, hidden),
#                                  nn.ReLU(),
#                                  nn.Linear(hidden, a_num))
#
#     def forward(self, s, h0=None, c0=None, return_hidden=True):
#         feature, (hn, cn) = self.lstm(s, (h0, c0))
#         output = self.net(feature)
#         if return_hidden:
#             return output, hn, cn
#         else:
#             return output

# s_dim, hidden, a_num, num_layers, trajectory_length = 2, 64, 4, 3, 5
# net = DRQ_Net(s_dim, hidden, a_num, num_layers)
#
# observation1 = torch.zeros(size=[trajectory_length, s_dim], dtype=torch.float)
# hn_0 = torch.zeros(size=[num_layers, hidden], dtype=torch.float)
# cn_0 = torch.zeros(size=[num_layers, hidden], dtype=torch.float)
#
# summary_, hn_, cn_ = net(observation1, hn_0, cn_0)
#
# print('no batch')
# print(observation1.shape)
# print(summary_.shape)
# print(hn_.shape)
# print(cn_.shape)
#
# print('batch')
#
#
# batch_size = 16
# trajectory_length = 5
# observation = torch.zeros(size=[batch_size, trajectory_length, s_dim])
# observation[-1, :, :] = 1
# print(observation.shape)
# hn_0 = torch.zeros(size=[num_layers, batch_size, hidden])
# cn_0 = torch.zeros(size=[num_layers, batch_size, hidden])
#
# summary_, hn_, cn_ = net(observation, hn_0, cn_0)
#
# print(observation)
# print(summary_)
#
# print(summary_.shape)
# print(hn_.shape)
# print(cn_.shape)