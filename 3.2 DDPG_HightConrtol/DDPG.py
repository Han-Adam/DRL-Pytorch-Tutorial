import json
import os
import torch
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np
from Network import Actor, Critic
from Memory import Memory


def _soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1. - tau) + param.data * tau
        )


class DDPG:
    def __init__(self,
                 path,
                 s_dim = 3,           # 状态空间维度,
                 a_dim = 1,            # 行动空间维度,
                 hidden = 64,          # 隐藏层宽度,
                 device = 'gpu',       # 训练位置,
                 capacity = 2e3,       # 记忆库大小
                 batch_size= 256,      # 训练批次大小,
                 start_lr_step = 512,  # 开始学习的时间
                 gamma=0.9,            # 回报折现率,
                 var_init = 1.,        # variance的初始值
                 var_decay = 0.9999,   # variance的衰减值
                 var_min = 0.1,        # variance的最小值
                 actor_lr = 1e-3,      # actor学习率,
                 critic_lr = 3e-4,     # critic学习率,
                 actor_tau = 0.1,      # actor更新率,
                 critic_tau = 0.2,     # critic更新率
    ):
        # 初始化所有需要的参数
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden = hidden
        # 因为我目前的测试机，无法使用gpu，所以gpu训练以后再加
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.capacity = capacity
        self.batch_size = batch_size
        self.start_lr_step = start_lr_step
        self.gamma = gamma
        self.var = var_init
        self.var_decay = var_decay
        self.var_min = var_min
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_tau = actor_tau
        self.critic_tau = critic_tau
        # 还没有使用
        self.path = path
        self.counter = 0

        # 初始化网络
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor_target = Actor(s_dim, a_dim, hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic = Critic(s_dim, a_dim, hidden)
        self.critic_target = Critic(s_dim, a_dim, hidden)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # 初始化记忆库
        self.memory = Memory(capacity, batch_size, self.device)

        # 是否继承以前的成果
        if not os.listdir(self.path + '/Net'):
            # 没有以前的东西可以继承
            print('init completed')
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            # 继承以前的网络与记忆
            print('loading completed')
            self.actor.load_state_dict(torch.load(self.path + '/Net/Actor.pth'))
            self.actor_target.load_state_dict(torch.load(self.path + '/Net/Actor_Target.pth'))
            self.critic.load_state_dict(torch.load(self.path + '/Net/Critic.pth'))
            self.critic_target.load_state_dict(torch.load(self.path + '/Net/Critic_Target.pth'))
            with open(self.path + '/Net/Memory.json', 'r') as f:
                self.memory.memory = json.load(f)
            with open(self.path + '/Net/Counter.json', 'r') as f:
                self.memory.counter = json.load(f)
            with open(self.path + '/Net/Var.json', 'r') as f:
                self.var = json.load(f)

    def choose_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            a = self.actor(s).numpy()
        a = np.clip(np.random.normal(loc=a, scale=self.var), -1., 1.)
        # 行动：仅为pitch_pos
        return a

    def store_transition(self, s, a, s_, r, done):
        # 向记忆库中存储经历
        self.memory.store_transition(s, a, s_, r, done)
        if self.memory.counter >= self.start_lr_step:
            s, a, s_, r, done = self.memory.get_sample()
            self._learn(s, a, s_, r, done)

    def store_network(self):
        # print('I stored actor in:', self.path+'/Net/Actor.pth')
        torch.save(self.actor.state_dict(), self.path + '/Net/Actor.pth')
        torch.save(self.actor_target.state_dict(), self.path + '/Net/Actor_Target.pth')
        torch.save(self.critic.state_dict(), self.path + '/Net/Critic.pth')
        torch.save(self.critic_target.state_dict(), self.path + '/Net/Critic_Target.pth')
        with open(self.path + '/Net/Memory.json', 'w') as f:
            json.dump(self.memory.memory, f)
        with open(self.path + '/Net/Counter.json', 'w') as f:
            json.dump(self.memory.counter, f)
        with open(self.path + '/Net/Var.json', 'w') as f:
            json.dump(self.var, f)

        print(self.var, self.memory.counter)

    def _learn(self, s, a, s_, r, done):
        # 更新critic
        td_target = r + (1-done) * self.gamma * self.critic_target(s_, self.actor_target(s_))
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, td_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 更新actor
        q = self.critic(s, self.actor(s))
        actor_loss = -torch.mean(q)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 更新target网络
        _soft_update(self.critic_target, self.critic, self.critic_tau)
        _soft_update(self.actor_target, self.actor, self.actor_tau)

        # update variance
        self.var = max(self.var * self.var_decay, self.var_min)
