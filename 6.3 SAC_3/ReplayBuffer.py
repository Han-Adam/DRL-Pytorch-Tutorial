import numpy as np
import torch
import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = [0]*capacity
        self.batch_size = batch_size
        self.counter = 0

    def store_transition(self, s, a, s_, r):
        index = self.counter % self.capacity
        self.memory[index] = [s, a, s_, r]
        self.counter += 1

    def get_sample(self):
        samples_indices = random.choices(range(min(self.capacity,self.counter)), k=self.batch_size)
        samples = np.array(self.memory)[samples_indices]
        s, a, s_, r = zip(* samples)

        s = torch.tensor(s, dtype=torch.float)    # [batch, s_dim]
        a = torch.tensor(a, dtype=torch.float)    # [batch, a_dim]
        s_ = torch.tensor(s_, dtype=torch.float)  # [batch, s_dim]
        r = torch.tensor(r, dtype=torch.float)    # [batch]
        r = torch.unsqueeze(r, dim=-1)            # [batch, 1]
        return s, a, s_, r