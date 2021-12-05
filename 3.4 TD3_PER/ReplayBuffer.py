import numpy as np
import torch
import random


class PER:
    def __init__(self, capacity, batch_size, alpha, beta):
        self.capacity = capacity
        self.memory = [0] * capacity
        self.priority = np.ones(shape=[capacity])
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        done = 0 if done is False else 1
        self.memory[index] = [s, a, s_, r, done]
        self.priority[index] = max(self.priority)
        self.counter += 1

    def get_sample(self):
        index = min(self.capacity, self.counter)
        # use priority to calculate probability
        priority = self.priority[0:index] ** self.alpha
        prob = priority / np.sum(priority)
        # calculate weight
        weight = (index * prob) ** (-self.beta)
        weight = weight / np.max(weight)
        # get samples
        samples_index = random.choices(range(index), k=self.batch_size, weights=prob)
        samples = np.array(self.memory)[samples_index]
        weight = weight[samples_index]
        s, a, s_, r, done = zip(*samples)

        s = torch.tensor(s, dtype=torch.float)            # [batch_size, s_dim]
        a = torch.tensor(a, dtype=torch.float)            # [batch_size, a_dim]
        s_ = torch.tensor(s_, dtype=torch.float)          # [batch_size, s_dim]
        r = torch.tensor(r, dtype=torch.float)            # [batch_size]
        r = torch.unsqueeze(r, dim=-1)                    # [batch_size, 1]
        done = torch.tensor(done, dtype=torch.float)      # [batch_size]
        done = torch.unsqueeze(done, dim=-1)              # [batch_size, 1]
        weight = torch.tensor(weight, dtype=torch.float)  # [batch_size]
        weight = torch.unsqueeze(weight, dim=-1)          # [batch_size, 1]
        return s, a, s_, r, done, weight, samples_index
