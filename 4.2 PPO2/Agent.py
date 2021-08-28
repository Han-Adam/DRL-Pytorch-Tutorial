import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from Network import Actor, Critic


class PPO:
    def __init__(self,
                 s_dim,
                 a_dim,
                 bound,
                 hidden,
                 device,
                 lr,
                 memory_len,
                 batch_size,
                 update_epoch,
                 gamma,
                 lambda_,
                 epsilon):
        # Parameter initialization
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.bound = bound
        self.hidden = hidden
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.memory_len = memory_len
        self.batch_size = batch_size
        self.update_epoch = update_epoch
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon

        # network initialization
        self.actor = Actor(s_dim, a_dim, hidden).to(self.device)
        self.actor_old = Actor(s_dim, a_dim, hidden).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(s_dim).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # memory initialization
        self.memory_s, self.memory_a, self.memory_r = [], [], []

    def get_action(self, s):
        # select action w.r.t the actions prob
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        mean, std = self.actor(s)
        cov = torch.diag_embed(std)
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        a = dist.sample()
        a = torch.clamp(a*self.bound, -self.bound, self.bound)
        # Because in this environment, action_dim equals 1, we use .item().
        # When action_dim>1, please yse .unmpy()
        return a.item()

    def learn(self, s, a, s_, r):
        # store transition
        self.memory_s.append(s)
        self.memory_a.append(a/self.bound)
        self.memory_r.append(r)
        if len(self.memory_r) == self.memory_len:
            # prepare of data
            s = torch.tensor(self.memory_s,
                             dtype=torch.float,
                             device=self.device)                           # [memory_len, s_dim]
            a = torch.tensor(self.memory_a,
                             dtype=torch.float,
                             device=self.device).unsqueeze(dim=-1)         # [memory_len, 1(a_dim)]
            r = torch.tensor(self.memory_r,
                             dtype=torch.float,
                             device=self.device)                           # [memory_len]
            s_ = torch.tensor(s_, dtype=torch.float, device=self.device)   # [s_dim]
            gae = self._gae(s, r, s_)
            r = self._discounted_r(r, torch.FloatTensor(s_))

            # calculate old log probability
            self.actor_old.load_state_dict(self.actor.state_dict())
            old_log_prob = self._log_prob(s, a, old=True)

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
            self.memory_s, self.memory_a, self.memory_r = [], [], []

    def _log_prob(self, s, a, old=False):
        # calculate the log probability
        if old:
            with torch.no_grad():
                mean, std = self.actor_old(s)
        else:
            mean, std = self.actor(s)

        cov = torch.diag_embed(std)
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
        log_prob = dist.log_prob(a).unsqueeze(dim=-1)
        return log_prob

    def _gae(self, s, r, s_):
        # calculate the general advantage estimation
        with torch.no_grad():
            v = self.critic(s).squeeze()        # [memory_len]
            v_ = self.critic(s_)                # [1]
            v_ = torch.cat((v[1:], v_))         # [memory_len]
            delta = r + self.gamma * v_ - v

            length = r.shape[0]
            gae = torch.zeros(size=[length])
            running_add = 0
            for t in range(length - 1, -1, -1):
                gae[t] = running_add * self.gamma * self.lambda_ + delta[t]
                running_add = gae[t]
            return torch.unsqueeze(gae, dim=-1)

    def _discounted_r(self, r, s_):
        # calculate the discounted reward
        with torch.no_grad():
            length = len(r)
            discounted_r = torch.zeros(size=[length])
            v_ = self.critic(s_).item()
            for t in range(length - 1, -1, -1):
                discounted_r[t] = v_ * self.gamma + r[t]
                v_ = discounted_r[t]
        return discounted_r.unsqueeze(dim=-1)

    def update_actor(self, s, a, gae, old_log_prob):
        # calculate the actor loss
        log_prob = self._log_prob(s, a)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio*gae
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * gae
        loss = -torch.mean(torch.min(surr1, surr2))
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