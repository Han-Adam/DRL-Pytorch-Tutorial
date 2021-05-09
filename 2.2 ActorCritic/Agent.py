import torch
from torch.distributions import Categorical
from Network import Actor, Critic


class ActorCritic:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            lr_actor,
            lr_critic,
            gamma,
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma

        # network initialization
        self.actor = Actor(s_dim, (s_dim+a_num)*3, a_num).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(s_dim, (s_dim+a_num)*3).to(self.device)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        # no memory in this algorithm

    def get_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        prob_weights = self.actor(s)
        # select action w.r.t the actions prob
        dist = Categorical(prob_weights)
        action = (dist.sample()).detach().item()
        return action

    def learn(self, s, a, s_, r, done):
        done = 1 if done else 0
        # torch.LongTensor torch.FloatTensor only work for list
        # when transform scalar to Tensor, we could use torch.tensor()
        s = torch.FloatTensor(s).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        r = torch.tensor(r, dtype=torch.float).to(self.device)
        # update for critic
        v = self.critic(s)
        v_ = self.critic(s_)
        td_target = r + (1-done)*self.gamma*v_.detach()
        td_error = td_target - v
        critic_loss = torch.pow(td_error, 2)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        # update for actor
        prob = self.actor(s)
        dist = Categorical(prob)
        actor_loss = -td_error.detach() * dist.log_prob(a)
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()