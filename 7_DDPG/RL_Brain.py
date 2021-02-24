import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RL_Brain:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 memory_capacity,
                 tau,
                 gamma,
                 lr_actor = 0.001,
                 lr_critic = 0.001,
                 batch_size=32,
                 ):
        self.tau = tau

        self.actor = Actor(n_input = state_dim,
                           n_hidden = state_dim*5,
                           n_output = action_dim,
                           bound = action_bound)
        self.critic = Critic(n_input_a = action_dim,
                             n_input_s = state_dim,
                             n_hidden = state_dim*5)
        self.actor_target = Actor(n_input = state_dim,
                                  n_hidden = state_dim*5,
                                  n_output = action_dim,
                                  bound = action_bound)
        self.critic_target = Critic(n_input_a = action_dim,
                                    n_input_s = state_dim,
                                    n_hidden = state_dim*5)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = Memory(state_dim= state_dim,
                             action_dim= action_dim,
                             batch_size= batch_size,
                             memory_capacity= memory_capacity)
        self.loss = Loss(actor = self.actor,
                         actor_target = self.actor_target,
                         critic = self.critic,
                         critic_target = self.critic_target,
                         gamma = gamma)

    def choose_action(self,s):
        action = self.actor(torch.tensor(s,dtype=torch.float))
        return action.detach().numpy()

    def store_transition(self, s, a, r, s_):
        self.memory.store_transition(s, a, r, s_)

    def learn(self):
        s,a,r,s_ = self.memory.get_sample()

        actor_loss = self.loss.actor_loss(s)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        critic_loss = self.loss.critic_loss(s, a, r, s_)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.soft_update(self.critic_target,self.critic)
        self.soft_update(self.actor_target,self.actor)

    def soft_update(self,target,source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


class Critic(nn.Module):
    def __init__(self,n_input_a,n_input_s,n_hidden):
        super(Critic,self).__init__()
        self.input_a = nn.Linear(n_input_a,n_hidden)
        self.input_a.weight.data.normal_(0, 0.1)
        self.input_s = nn.Linear(n_input_s,n_hidden)
        self.input_s.weight.data.normal_(0, 0.1)
        self.output = nn.Linear(n_hidden,1)

    def forward(self,s,a):
        input_a = self.input_a(a)
        input_s = self.input_s(s)
        input = F.relu(input_a+input_s)
        v = self.output(input)
        return v


class Actor(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,bound):
        super(Actor,self).__init__()
        self.bound = torch.tensor(bound, dtype=torch.float)
        self.input = nn.Linear(n_input,n_hidden)
        self.input.weight.data.normal_(0, 0.1)
        self.output = nn.Linear(n_hidden,n_output)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self,s):
        input = F.relu(self.input(s))
        action = F.tanh(self.output(input))
        action = action*self.bound
        return action

class Memory:
    def __init__(self, state_dim, action_dim, batch_size, memory_capacity):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory_s = torch.tensor([[0] * state_dim for _ in range(memory_capacity)], dtype=torch.float)
        self.memory_a = torch.tensor([[0] * action_dim for _ in range(memory_capacity)], dtype=torch.float)
        self.memory_r = torch.tensor([[0] * 1 for _ in range(memory_capacity)], dtype=torch.float)
        self.memory_s_ = torch.tensor([[0] * state_dim for _ in range(memory_capacity)], dtype=torch.float)

        self.counter = 0

    def store_transition(self, s, a, r, s_):
        index = self.counter % self.memory_capacity
        self.memory_s[index] = torch.tensor(s, dtype=torch.float)
        self.memory_a[index] = torch.tensor(a, dtype=torch.float)
        self.memory_r[index] = torch.tensor(r, dtype=torch.float)
        self.memory_s_[index] = torch.tensor(s_, dtype=torch.float)
        self.counter += 1

    def get_sample(self):
        sample_index = np.random.choice(
            a=min(self.memory_capacity, self.counter),
            size=self.batch_size)

        s = self.memory_s[sample_index]
        a = self.memory_a[sample_index]
        r = self.memory_r[sample_index]
        s_ = self.memory_s_[sample_index]
        return s, a, r, s_

class Loss:
    def __init__(self, actor, actor_target, critic, critic_target, gamma):
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = gamma

    def actor_loss(self,s):
        q_value = self.critic(s, self.actor(s))
        actor_loss = -torch.mean(q_value)
        return actor_loss

    def critic_loss(self,s,a,r,s_):
        td_target = r + self.gamma * self.critic_target(s_, self.actor_target(s_))
        q_value = self.critic(s, a)
        critic_loss = F.mse_loss(q_value, td_target)
        return critic_loss
