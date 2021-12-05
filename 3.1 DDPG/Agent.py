import torch
import numpy as np
import torch.nn.functional as F
from Network import Actor, Critic
from ReplayBuffer import ReplayBuffer


class DDPG:
    def __init__(
            self,
            s_dim,
            a_dim,
            device,
            hidden,
            capacity,
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
        self.device = device
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
        self.actor = Actor(s_dim, hidden, a_dim).to(device)
        self.actor_target = Actor(s_dim, hidden, a_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(s_dim, a_dim, hidden).to(device)
        self.critic_target = Critic(s_dim, a_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # replay buffer, or memory
        self.memory = ReplayBuffer(capacity, batch_size, device)

    def get_action(self, s):
        with torch.no_grad():
            s = torch.FloatTensor(s).to(self.device)
            a = self.actor(s).numpy()
        a = np.clip(np.random.normal(a, self.var), -1., 1.)
        return a

    def learn(self):
        # samples from memory
        s, a, s_, r, done = self.memory.get_sample()

        # update critic
        with torch.no_grad():
            td_target = r + (1-done)*self.gamma*self.critic_target(s_, self.actor_target(s_))
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
