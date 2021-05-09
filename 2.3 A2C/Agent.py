import torch
from torch.distributions import Categorical
from Network import Actor, Critic


class A2C:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            lr_actor,
            lr_critic,
            max_len,
            gamma,
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.max_len = max_len
        self.gamma = gamma

        # network initialization
        self.actor = Actor(s_dim, (s_dim+a_num)*3, a_num).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(s_dim, (s_dim+a_num)*3).to(self.device)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # define memory
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

    def get_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        prob_weights = self.actor(s)
        # select action w.r.t the actions prob
        dist = Categorical(prob_weights)
        action = (dist.sample()).detach().item()
        return action

    def learn(self, s, a, s_, r, done):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)
        if len(self.memory_r)>= self.max_len or done:
            # calculate the discounted reward
            discounted_r = []
            value_ = 0 if done else self.critic(torch.FloatTensor(s_).to(self.device)).item()
            for t in range(len(self.memory_r) - 1, -1, -1):
                value_ = value_ * self.gamma + self.memory_r[t]
                discounted_r.insert(0, value_)
            # start to learn
            s = torch.FloatTensor(self.memory_s).to(self.device)
            a = torch.LongTensor(self.memory_a).to(self.device)
            r = torch.FloatTensor(discounted_r).to(self.device)
            # update critic
            v = self.critic(s)
            advantage = r - v
            critic_loss = torch.mean(torch.pow(advantage, 2))
            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()
            # update actor
            probs = self.actor(s)
            dist = Categorical(probs)
            log_probs = dist.log_prob(a)
            actor_loss = -torch.mean(log_probs * advantage.detach())
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()
            # renew the memory
            self.memory_s = []
            self.memory_a = []
            self.memory_r = []