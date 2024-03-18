import numpy as np
import torch
import random


class PER:
    def __init__(self, s_dim, a_dim, capacity, batch_size, rank):
        self.capacity = capacity
        self.s = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False)
        self.a = torch.zeros(size=[capacity, a_dim], dtype=torch.float, requires_grad=False)
        self.s_ = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False)
        self.r = torch.zeros(size=[capacity, 1], dtype=torch.float, requires_grad=False)
        self.done = torch.zeros(size=[capacity, 1], dtype=torch.float, requires_grad=False)
        self.priority = np.ones(shape=[self.capacity])
        self.batch_size = batch_size
        if rank:
            self.rank, self.alpha, self.beta = True, 0.6, 0.4
        else:
            self.rank, self.alpha, self.beta = False, 0.7, 0.5
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        self.s[index] = torch.tensor(s, dtype=torch.float)
        self.a[index] = torch.tensor(a, dtype=torch.long)
        self.s_[index] = torch.tensor(s_, dtype=torch.float)
        self.r[index] = torch.tensor(r, dtype=torch.float)
        self.done[index] = torch.tensor(1 if done else 0, dtype=torch.long)
        self.priority[index] = np.max(self.priority)
        self.counter += 1

    def get_sample(self):
        index = min(self.capacity, self.counter)
        priority = self.priority[0:index]
        # use priority to calculate probability
        if self.rank:
            # use rank version priority
            sort = np.argsort(priority)
            rank = np.zeros(shape=[index])
            for i in range(index):
                rank[sort[i]] = i + 1
            priority = (1 / rank)
        priority = priority ** self.alpha
        prob = priority / np.sum(priority)
        # calculate weight
        weight = (index * prob) ** (-self.beta)
        weight = weight / np.max(weight)
        # get samples
        samples_index = random.choices(range(index), k=self.batch_size, weights=prob)
        s = self.s[samples_index]        # [batch, s_dim]
        a = self.a[samples_index]        # [batch, a_dim]
        s_ = self.s_[samples_index]      # [batch, s_dim]
        r = self.r[samples_index]        # [batch, 1]
        done = self.done[samples_index]  # [batch, 1]
        weight = torch.tensor(weight[samples_index]) # [batch]
        # s, a, s_, r, done = zip(*samples)

        # s = torch.tensor(s, dtype=torch.float)            # [batch_size, s_dim]
        # a = torch.tensor(a, dtype=torch.float)            # [batch_size, a_dim]
        # s_ = torch.tensor(s_, dtype=torch.float)          # [batch_size, s_dim]
        # r = torch.tensor(r, dtype=torch.float)            # [batch_size]
        # r = torch.unsqueeze(r, dim=-1)                    # [batch_size, 1]
        # done = torch.tensor(done, dtype=torch.float)      # [batch_size]
        # done = torch.unsqueeze(done, dim=-1)              # [batch_size, 1]
        # weight = torch.tensor(weight, dtype=torch.float)  # [batch_size]
        # weight = torch.unsqueeze(weight, dim=-1)          # [batch_size, 1]
        return s, a, s_, r, done, weight, samples_index
