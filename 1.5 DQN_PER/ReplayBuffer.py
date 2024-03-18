import numpy as np
import torch
import random


class PrioritizedReplayBuffer:
    def __init__(self, s_dim, capacity, batch_size, rank):
        self.capacity = capacity
        self.s = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False)
        self.a = torch.zeros(size=[capacity], dtype=torch.long, requires_grad=False)
        self.s_ = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False)
        self.r = torch.zeros(size=[capacity], dtype=torch.float, requires_grad=False)
        self.done = torch.zeros(size=[capacity], dtype=torch.float, requires_grad=False)
        self.priority = np.ones(shape=[self.capacity])
        self.batch_size = batch_size
        # whether we use rank or proportional version
        # the setting of alpha/beta follows the original paper
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
        self.priority[index] = max(self.priority)
        self.counter += 1

    def get_sample(self):
        index = min(self.capacity, self.counter)
        priority = self.priority[0:index]
        if self.rank:
            # use rank version priority
            sort = np.argsort(priority)
            rank = np.zeros(shape=[index])
            for i in range(index):
                rank[sort[i]] = i + 1
            priority = (1 / rank)
        # use priority to calculate probability
        priority = priority**self.alpha
        prob = priority/np.sum(priority)
        # calculate weight
        weight = (index*prob)**(-self.beta)
        weight = weight/np.max(weight)
        # get samples
        samples_index = random.choices(range(index), k=self.batch_size, weights=prob)
        s = self.s[samples_index]  # [batch, s_dim]
        a = self.a[samples_index]  # [batch]
        s_ = self.s_[samples_index]  # [batch, s_dim]
        r = self.r[samples_index]  # [batch]
        done = self.done[samples_index]  # [batch]
        weight = torch.FloatTensor(weight[samples_index])



        # samples = np.array(self.memory)[samples_index]
        # weight = weight[samples_index]
        # s, a, s_, r, done = zip(* samples)
        #
        # s = torch.FloatTensor(s)
        # a = torch.LongTensor(a)
        # s_ = torch.FloatTensor(s_)
        # r = torch.FloatTensor(r)
        # done = torch.FloatTensor(done)
        # weight = torch.FloatTensor(weight)
        return s, a, s_, r, done, weight, samples_index