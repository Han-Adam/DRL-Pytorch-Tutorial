import torch
import torch.nn as nn


def init_linear(module):
    # 用于初始化网络参数
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)


class Actor(nn.Module):
    # 定义三隐藏层MLP，输入是状态，输出是行动
    def __init__(self, s_dim, a_dim, hidden):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, a_dim),
                                   nn.Tanh())
        init_linear(self)

    def forward(self, s):
        return self.actor(s)


class Critic(nn.Module):
    # 定义三隐藏层MLP，V网络，输入为状态，输出是期望折现回报
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(s_dim+a_dim, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, 1))
        init_linear(self)

    def forward(self, s, a):
        return self.critic(torch.cat([s, a], dim=-1))

