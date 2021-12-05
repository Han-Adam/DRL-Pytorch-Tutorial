import numpy as np
import torch
import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size, device, rank):
        self.capacity = capacity
        self.memory = [0]*self.capacity
        self.priority = np.ones(shape=[self.capacity])
        self.batch_size = batch_size
        self.device = device
        # whether we use rank or proportional version
        # the setting of alpha/beta follows the original paper
        if rank:
            self.rank, self.alpha, self.beta = True, 0.6, 0.4
        else:
            self.rank, self.alpha, self.beta = False, 0.7, 0.5
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        done = 0 if done is False else 1
        self.memory[index] = [s, a, s_, r, done]
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
        samples = np.array(self.memory)[samples_index]
        weight = weight[samples_index]
        s, a, s_, r, done = zip(* samples)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weight = torch.FloatTensor(weight).to(self.device)
        return s, a, s_, r, done, weight, samples_index