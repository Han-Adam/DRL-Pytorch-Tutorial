import torch
import numpy as np
import torch.nn.functional as F
from Network import QNet, PolicyNet
from ReplayBuffer import ReplayBuffer
from torch.distributions import Normal

class SAC:
    def __init__(
            self,
            s_dim,
            a_dim,
            bound,
            device,
            capacity,
            batch_size,
            lr,
            gamma,
            tau,
            log_prob_reg
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.bound = bound
        self.device = device
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.log_prob_reg = log_prob_reg

        hidden = 256
        # Network
        self.q_net = QNet(s_dim, a_dim, hidden).to(device)
        self.target_q_net = QNet(s_dim, a_dim, hidden).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_net = PolicyNet(s_dim, a_dim, hidden).to(device)
        self.opt_policy = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        # alpha
        self.alpha = 1
        self.target_entropy = -a_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)


        # replay buffer, memory
        self.memory = ReplayBuffer(capacity, batch_size, device)

    def get_action(self, s):
        s = torch.tensor(data=s, dtype=torch.float, device=self.device)
        mean, std = self.policy_net(s)

        normal = Normal(mean, std)
        z = normal.rsample()
        a = torch.tanh(z)

        return self.bound*a.detach().item()

    def get_logprob(self, s, log_reg=1e-6):
        mean, std = self.policy_net(s)

        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)

        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + log_reg)
        log_prob = log_prob.sum(-1, keepdim=True)

        return a, log_prob

    def learn(self):
        # samples from memory
        s, a, s_, r, done = self.memory.get_sample()

        # update q net
        q = self.q_net(s, a)
        a_, log_prob_ = self.get_logprob(s_)
        q_ = self.target_q_net(s_, a_)
        q_target = r + (1 - done) * self.gamma * (q_ - self.alpha*log_prob_)
        q_loss = F.mse_loss(q, q_target.detach())

        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        # update policy net
        a_new, log_prob_new = self.get_logprob(s)
        q_new = self.q_net(s, a_new)
        # both loss_functions are available
        # policy_loss = F.mse_loss(log_prob, q_new)
        policy_loss = torch.mean(self.alpha*log_prob_new - q_new)

        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # update temperature alpha
        alpha_loss = -torch.mean(self.log_alpha* (log_prob_new+self.target_entropy).detach())
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        # update target net
        self.soft_update(self.target_q_net, self.q_net)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )