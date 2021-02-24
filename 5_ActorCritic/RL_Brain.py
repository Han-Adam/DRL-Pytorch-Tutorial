import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RL_Brain:
    def __init__(
            self,
            n_actions,
            n_features,
            lr_actor=0.01,
            lr_critic=0.01,
            reward_decay=0.9,
    ):
        self.n_action = n_actions
        self.n_feature = n_features
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = reward_decay

        self.actor = Actor(self.n_feature,self.n_feature*2,self.n_action)
        self.critic = Critic(self.n_feature,self.n_feature*2)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=self.lr_critic)
        self.loss = Loss(self.actor,self.critic,self.gamma)

    def choose_action(self,s):
        prob = self.actor(torch.tensor(s,dtype=torch.float))
        dist = Categorical(prob)
        action = dist.sample().detach().numpy()
        return action

    def learn(self, s, a, s_, r, done):
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        s_ = torch.tensor(s_, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)
        critic_loss, actor_loss = self.loss(s, a, s_, r, done)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()


class Critic(nn.Module):
    # Critic Network
    def __init__(self,n_input,n_hidden):
        super(Critic,self).__init__()
        self.input = nn.Linear(n_input,n_hidden)
        self.output = nn.Linear(n_hidden,1)

    def forward(self, s):
        input = F.relu(self.input(s))
        v = self.output(input)
        return v


class Actor(nn.Module):
    # Actor Network
    def __init__(self,n_input,n_hidden,n_output):
        super(Actor,self).__init__()
        self.input = nn.Linear(n_input,n_hidden)
        self.output = nn.Linear(n_hidden,n_output)

    def forward(self,s):
        input = F.relu(self.input(s))
        probability = F.softmax(self.output(input))
        return probability


class Loss(nn.Module):
    # Loss calculation
    def __init__(self,actor,critic,gamma):
        super(Loss,self).__init__()
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def forward(self, s, a, s_, r, done):
        # critic loss calculation
        v = self.critic(s)
        v_ = self.critic(s_)
        td_error = (r + self.gamma*v_- v).detach()
        #critic_loss = -td_error*v
        critic_loss = torch.pow((r + self.gamma*v_.detach()- v),2)
        # 这里critic_loss和actor_loss 正负号相同就可以收敛, why?
        # actor loss calculation
        prob = self.actor(torch.tensor(s, dtype=torch.float))
        dist = Categorical(prob)
        actor_loss = -td_error*dist.log_prob(a)

        return critic_loss, actor_loss