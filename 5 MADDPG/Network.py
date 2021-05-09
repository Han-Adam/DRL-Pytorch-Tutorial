import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    # centralized critic
    # input: state and action for all agents
    # output: Q_value for one agent
    def __init__(self, state_n, action_n, hidden):
        super(Critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_1 = nn.Linear(state_n + action_n, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, s, a):
        x_cat = self.LReLU(self.linear_1(torch.cat([s, a], dim=1)))
        x = self.LReLU(self.linear_2(x_cat))
        value = self.linear_3(x)
        return value


class Actor(nn.Module):
    # decentralized critic
    # input: state and action of one agent
    # output: policy for one agent
    def __init__(self, state_n, action_n, hidden):
        super(Actor, self).__init__()
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_1 = nn.Linear(state_n, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, action_n)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, s, model_original_out=False):
        x = self.LReLU(self.linear_1(s))
        x = self.LReLU(self.linear_2(x))
        # model_out: the original deterministic policy
        model_out = self.linear_3(x)
        u = torch.rand_like(model_out)
        # policy: consider the random factor, use for exploration.
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out == True:
            return model_out, policy
        else:
            return policy
