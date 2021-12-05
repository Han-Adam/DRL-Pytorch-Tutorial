import os
import json
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from Network import Actor, Critic


class PPO:
    def __init__(self,
                 path,
                 s_dim=3,
                 a_dim=1,
                 hidden=64,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 memory_len=64,
                 batch_size=32,
                 update_epoch=10,
                 gamma=0.9,
                 lambda_=0.95,
                 epsilon=0.2):
        # Parameter initialization
        self.path = path
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden = hidden
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.memory_len = memory_len
        self.batch_size = batch_size
        self.update_epoch = update_epoch
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon

        # network initialization
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor_old = Actor(s_dim, a_dim, hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic = Critic(s_dim, hidden)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # memory initialization
        self.memory_s, self.memory_a, self.memory_s_, self.memory_r, self.memory_done = [], [], [], [], []

        # 是否继承以前的成果
        if not os.listdir(self.path + '/Net'):
            # 没有以前的东西可以继承
            print('init completed')
        else:
            # 继承以前的网络与记忆
            print('loading completed')
            self.actor.load_state_dict(torch.load(self.path + '/Net/Actor.pth'))
            self.critic.load_state_dict(torch.load(self.path + '/Net/Critic.pth'))
            with open(self.path + '/Net/Memory_s.json', 'r') as f:
                self.memory_s = json.load(f)
            with open(self.path + '/Net/Memory_a.json', 'r') as f:
                self.memory_a = json.load(f)
            with open(self.path + '/Net/Memory_s_.json', 'r') as f:
                self.memory_s_ = json.load(f)
            with open(self.path + '/Net/Memory_r.json', 'r') as f:
                self.memory_r = json.load(f)
            with open(self.path + '/Net/Memory_done.json', 'r') as f:
                self.memory_done = json.load(f)
        self.actor_old.load_state_dict(self.actor.state_dict())

    def store_network(self):
        torch.save(self.actor.state_dict(), self.path + '/Net/Actor.pth')
        torch.save(self.critic.state_dict(), self.path + '/Net/Critic.pth')
        with open(self.path + '/Net/Memory_s.json', 'w') as f:
            json.dump(self.memory_s, f)
        with open(self.path + '/Net/Memory_a.json', 'w') as f:
            json.dump(self.memory_a, f)
        with open(self.path + '/Net/Memory_s_.json', 'w') as f:
            json.dump(self.memory_s_, f)
        with open(self.path + '/Net/Memory_r.json', 'w') as f:
            json.dump(self.memory_r, f)
        with open(self.path + '/Net/Memory_done.json', 'w') as f:
            json.dump(self.memory_done, f)

    def choose_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            mean, std = self.actor(s)
            cov = torch.diag_embed(std)
            dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
            a = dist.sample()
            a = torch.clamp(a, -1., 1.).numpy().tolist()
        return a

    def store_transition(self, s, a, s_, r, done):
        # store transition
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_s_.append(s_)
        self.memory_r.append(r)
        self.memory_done.append(1 if done else 0)
        if len(self.memory_r) == self.memory_len:
            # prepare of data
            s = torch.tensor(self.memory_s,
                             dtype=torch.float)    # [memory_len, s_dim]
            a = torch.tensor(self.memory_a,
                             dtype=torch.float)    # [memory_len, 1(a_dim)]
            r = torch.tensor(self.memory_r,
                             dtype=torch.float)    # [memory_len]
            s_ = torch.tensor(self.memory_s_,
                              dtype=torch.float)   # [memory_len, s_dim]
            done = torch.tensor(self.memory_done,
                                dtype=torch.float) # [memory_len]
            self._learn(s, a, s_, r, done)

    def _learn(self, s, a, s_, r, done):
        gae = self._gae(s, r, s_, done)            # [memory_len, 1]
        r = self._discounted_r(r, s_, done)        # [memory_len, 1]

        # calculate old log probability
        self.actor_old.load_state_dict(self.actor.state_dict())
        old_log_prob = self._log_prob(s, a, old=True)                  # [memory_len, 1]

        # batch update the network
        for i in range(self.update_epoch):
            for index in range(0, self.memory_len, self.batch_size):
                self.update_actor(s[index: index+self.batch_size],
                                  a[index: index+self.batch_size],
                                  gae[index: index+self.batch_size],
                                  old_log_prob[index: index+self.batch_size])
                self.update_critic(s[index: index+self.batch_size],
                                   r[index: index+self.batch_size])
        # empty the memory
        self.memory_s, self.memory_a, self.memory_s_, self.memory_r, self.memory_done = [], [], [], [], []

    def _log_prob(self, s, a, old=False):
        # calculate the log probability
        if old:
            with torch.no_grad():
                mean, std = self.actor_old(s)
        else:
            mean, std = self.actor(s)
        std = torch.stack([std]*mean.shape[0], dim=0)

        cov = torch.diag_embed(std)
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        log_prob = dist.log_prob(a).unsqueeze(dim=-1)
        return log_prob

    def _gae(self, s, r, s_, done):
        # calculate the general advantage estimation
        with torch.no_grad():
            v = self.critic(s).squeeze()        # [memory_len]
            v_ = self.critic(s_).squeeze()      # [memory_len]
            delta = r + self.gamma * v_ - v

            length = r.shape[0]
            gae = torch.zeros(size=[length])
            running_add = 0
            for t in range(length - 1, -1, -1):
                gae[t] = running_add * self.gamma * self.lambda_ * (1-done[t]) + delta[t]
                running_add = gae[t]
            return torch.unsqueeze(gae, dim=-1)

    def _discounted_r(self, r, s_, done):
        # calculate the discounted reward
        with torch.no_grad():
            length = len(r)
            discounted_r = torch.zeros(size=[length])
            v_ = self.critic(s_)
            running_add = 0
            for t in range(length - 1, -1, -1):
                if done[t] == 1 or t == length-1:
                    discounted_r[t] = v_[t] * self.gamma + r[t]
                else:
                    discounted_r[t] = running_add * self.gamma + r[t]
                running_add = discounted_r[t]
        return discounted_r.unsqueeze(dim=-1)

    def update_actor(self, s, a, gae, old_log_prob):
        # calculate the actor loss
        log_prob = self._log_prob(s, a)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio*gae
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * gae
        loss = -torch.mean(torch.min(surr1, surr2))
        loss = loss - 0.001*self.actor.log_std   # 这个任务当中，加入PPO是有效果的。
        # update
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def update_critic(self, s, r):
        # calculate critic loss
        v = self.critic(s)
        loss = F.mse_loss(v, r)
        # update
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()