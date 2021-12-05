import copy
import torch
import numpy as np
import torch.nn.functional as F
from Network import Actor, Critic
from ReplayBuffer import PER


class TD3:
    def __init__(self,
                 s_dim,
                 a_dim,
                 capacity,
                 batch_size,
                 lr_actor,
                 lr_critic,
                 alpha,
                 beta,
                 p_with_pi,
                 hidden,
                 reg_coe,
                 var_init,
                 var_decay,
                 var_min,
                 gamma,
                 tau,
                 policy_noise,
                 noise_clip,
                 policy_freq):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.alpha = alpha
        self.beta = beta
        self.p_with_pi = p_with_pi
        self.hidden = hidden
        self.reg_coe = reg_coe
        self.capacity = capacity
        self.batch_size = batch_size
        self.var = var_init
        self.var_decay = var_decay
        self.var_min = var_min
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.train_it = 0

        # Network
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor_target = copy.deepcopy(self.actor)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=reg_coe)

        self.critic = Critic(s_dim, a_dim, hidden)
        self.critic_target = copy.deepcopy(self.critic)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=reg_coe)

        # replay buffer, or memory
        self.memory = PER(capacity, batch_size, alpha, beta)

    def get_action(self, s):
        with torch.no_grad():
            a = self.actor(torch.FloatTensor(s))
        #  add randomness to action selection for exploration
        a = a.numpy()
        a = np.clip(np.random.normal(a, self.var), -1., 1.)
        return a

    def learn(self):
        self.train_it += 1
        s, a, s_, r, done, weight, samples_index = self.memory.get_sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.clip(torch.randn_like(a) * self.policy_noise,
                               -self.noise_clip, self.noise_clip)
            a_ = torch.clip(self.actor_target(s_)+noise, -1., 1.)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(s_, a_)
            target_Q = torch.min(target_Q1, target_Q2)
            td_target = r + (1-done) * self.gamma * target_Q

        # update critic
        q1, q2 = self.critic(s, a)
        td_error = (q1 - td_target)**2 + (q2 - td_target)**2
        # critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        critic_loss = torch.mean(td_error)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        if not self.p_with_pi:
            new_priority = torch.abs(td_error.squeeze()).detach().numpy() + \
                           (np.e ** -10)  # + (np.e ** -10))**self.memory.alpha
            self.memory.priority[samples_index] = new_priority

        if self.train_it % self.policy_freq == 0:
            # update actor
            q = self.critic.Q1(s, self.actor(s))
            actor_loss = -torch.mean(q)
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()
            if self.p_with_pi:
                new_priority = torch.abs(td_error.squeeze()).detach().numpy() + \
                               torch.pow(q.squeeze(), 2).detach().numpy() + \
                               (np.e ** -10)  # + (np.e ** -10))**self.memory.alpha
                self.memory.priority[samples_index] = new_priority

            # update target network
            self.soft_update(self.critic_target, self.critic)
            self.soft_update(self.actor_target, self.actor)

            # update varaiance
            self.var = max(self.var * self.var_decay, self.var_min)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
