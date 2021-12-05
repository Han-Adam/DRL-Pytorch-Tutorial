import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from Network import Policy, QNet, VNet, Discriminator
from ReplayBuffer import ReplayBuffer


class DIAYN:
    def __init__(self,
                 s_dim,
                 a_num,
                 skill_num,
                 hidden,
                 lr,
                 gamma,
                 tau,
                 log_prob_reg,
                 alpha,
                 capacity,
                 batch_size,
                 device
                 ):
        self.s_dim = s_dim
        self.a_num = a_num
        self.skill_num = skill_num
        hidden = hidden
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.log_prob_reg = log_prob_reg
        self.alpha = alpha
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.log_pz = torch.log(torch.tensor(1/skill_num, dtype=torch.float, device=device))

        # network initialization
        self.policy = Policy(s_dim, skill_num, hidden, a_num).to(device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.q_net = QNet(s_dim, skill_num, hidden, a_num).to(device)
        self.opt_q_net = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.v_net = VNet(s_dim, skill_num, hidden).to(device)
        self.v_net_target = VNet(s_dim, skill_num, hidden).to(device)
        self.v_net_target.load_state_dict(self.v_net.state_dict())
        self.opt_v_net = torch.optim.Adam(self.v_net.parameters(), lr=lr)

        self.discriminator = Discriminator(s_dim, skill_num, hidden).to(device)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # replay buffer, memory
        self.memory = ReplayBuffer(capacity, batch_size, device)

    def get_action(self, s, z):
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        z = torch.tensor(z, dtype=torch.float, device=self.device)
        prob = self.policy(s, z)
        dist = Categorical(prob)
        a = dist.sample()
        return a.item()

    def get_pseudo_reward(self, s, z, a, s_):
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        z = torch.tensor(z, dtype=torch.float, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        s_ = torch.tensor(s_, dtype=torch.float, device=self.device)

        pseudo_reward = self.discriminator(s_,log=True)[z.argmax(dim=-1)] - \
                        self.log_pz + \
                        self.alpha*self.policy(s,z)[a]

        return pseudo_reward.detach().item()


    def learn(self):
        index = torch.tensor(range(self.batch_size), dtype=torch.long, device=self.device)
        s, z, a, s_, r, done = self.memory.get_sample()
        # soft-actor-critic update
        # update q net
        q = self.q_net(s, z)[index, a].unsqueeze(dim=-1)
        v_ = self.v_net_target(s_, z)
        q_target = r + (1 - done) * self.gamma * v_
        q_loss = F.mse_loss(q, q_target.detach())

        self.opt_q_net.zero_grad()
        q_loss.backward()
        self.opt_q_net.step()

        # update v net
        v = self.v_net(s, z)
        log_prob = self.policy(s, z, log=True)[index,a].unsqueeze(dim=-1)
        q_new = self.q_net(s, z)[index, a].unsqueeze(dim=-1)
        v_target = q_new - log_prob
        v_loss = F.mse_loss(v, v_target.detach())

        self.opt_v_net.zero_grad()
        v_loss.backward()
        self.opt_v_net.step()

        # update policy net
        policy_loss = F.mse_loss(log_prob, q_new.detach())
        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # update target net
        self.soft_update(self.v_net_target, self.v_net)

        # update discriminator
        log_q_zs = self.discriminator(s,log=True)
        discriminator_loss = F.nll_loss(log_q_zs, z.argmax(dim=-1))
        self.opt_discriminator.zero_grad()
        discriminator_loss.backward()
        self.opt_discriminator.step()


    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )