import torch
import random


class ReplayBuffer:
    def __init__(self, s_dim, capacity, batch_size):
        self.capacity = capacity
        self.s = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False)
        self.a = torch.zeros(size=[capacity], dtype=torch.long, requires_grad=False)
        self.s_ = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False)
        self.r = torch.zeros(size=[capacity], dtype=torch.float, requires_grad=False)
        self.done = torch.zeros(size=[capacity], dtype=torch.float, requires_grad=False)
        self.batch_size = batch_size
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        self.s[index] = torch.tensor(s, dtype=torch.float)
        self.a[index] = torch.tensor(a, dtype=torch.long)
        self.s_[index] = torch.tensor(s_, dtype=torch.float)
        self.r[index] = torch.tensor(r, dtype=torch.float)
        self.done[index] = torch.tensor(1 if done else 0, dtype=torch.long)
        self.counter += 1

    def get_sample(self):
        index = random.choices(range(min(self.capacity, self.counter)), k=self.batch_size)
        s = self.s[index]              # [batch, s_dim]
        a = self.a[index]              # [batch]
        s_ = self.s_[index]            # [batch, s_dim]
        r = self.r[index]              # [batch]
        done = self.done[index]        # [batch]
        return s, a, s_, r, done