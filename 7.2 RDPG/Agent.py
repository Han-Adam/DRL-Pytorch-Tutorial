import torch
import numpy as np
import torch.nn.functional as F
from Network import Actor, Critic
from ReplayBuffer import ReplayBuffer


class RDPG:
    def __init__(
            self,
            s_dim,
            a_dim,
            hidden,
            rnn_num_layers,
            capacity,
            max_trajectory_len,
            batch_size,
            lr_actor,
            lr_critic,
            variance_start,
            variance_decay,
            variance_min,
            gamma,
            tau
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden = hidden
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.capacity = capacity
        self.batch_size = batch_size
        self.var = variance_start
        self.var_decay = variance_decay
        self.var_min = variance_min
        self.gamma = gamma
        self.tau = tau

        # Network
        self.actor = Actor(s_dim, a_dim, hidden, rnn_num_layers)
        self.actor_target = Actor(s_dim, a_dim, hidden, rnn_num_layers)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(s_dim, a_dim, hidden, rnn_num_layers)
        self.critic_target = Critic(s_dim, a_dim, hidden, rnn_num_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.hn = torch.zeros(size=[rnn_num_layers, hidden], dtype=torch.float)
        self.cn = torch.zeros(size=[rnn_num_layers, hidden], dtype=torch.float)

        # experience replay buffer, memory
        self.memory = ReplayBuffer(s_dim, a_dim, capacity, max_trajectory_len, batch_size)
        self.counter = 0

    def get_action(self, s):
        with torch.no_grad():
            s = torch.unsqueeze(torch.FloatTensor(s), dim=0)
            a, self.hn, self.cn = self.actor(s, self.hn, self.cn, True)
            a = torch.squeeze(a, dim=0).numpy()
        a = np.clip(np.random.normal(a, self.var), -1., 1.)
        return a

    def learn(self):
        # samples from memory
        s, a, s_, r, done = self.memory.get_sample()
        with torch.no_grad():
            a_ = self.actor_target(s_)
            td_target = r + (1-done)*self.gamma*self.critic_target(s_, a_)
        q = self.critic(s, a)

        critic_loss = F.mse_loss(q, td_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # update actor
        q = self.critic(s, self.actor(s))
        actor_loss = -torch.mean(q)
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # update target network
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

        # update variance
        self.var = max(self.var*self.var_decay, self.var_min)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
