import numpy as np
import torch
import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size, device):
        self.capacity = capacity
        self.memory = [0]*capacity
        self.batch_size = batch_size
        self.device = device
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        done = 0 if done is False else 1
        self.memory[index] = [s, a, s_, r, done]
        self.counter += 1

    def get_sample(self):
        samples_indices = random.choices(range(min(self.capacity, self.counter)), k=self.batch_size)
        samples = np.array(self.memory)[samples_indices]
        s, a, s_, r, done = zip(* samples)

        s = torch.FloatTensor(s).to(self.device)       # [batch, s_dim]
        a = torch.LongTensor(a).to(self.device)        # [batch, a_dim]
        s_ = torch.FloatTensor(s_).to(self.device)     # [batch, s_dim]
        r = torch.FloatTensor(r).to(self.device)       # [batch]
        r = torch.unsqueeze(r, dim=-1)                 # [batch, 1]
        done = torch.FloatTensor(done).to(self.device) # [batch]
        done = torch.unsqueeze(done, dim=-1)           # [batch, 1]

        return s, a, s_, r, done