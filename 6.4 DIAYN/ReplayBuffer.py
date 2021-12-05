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

    def store_transition(self, s, z, a, s_, r, done):
        index = self.counter % self.capacity
        done = 0 if done is False else 1
        self.memory[index] = [s, z, a, s_, r, done]
        self.counter += 1

    def get_sample(self):
        samples_indices = random.choices(range(min(self.capacity,self.counter)), k=self.batch_size)
        samples = np.array(self.memory)[samples_indices]
        s, z, a, s_, r, done = zip(* samples)

        s = torch.tensor(s, dtype=torch.float, device=self.device)
        z = torch.tensor(z, dtype=torch.float, device=self.device)
        a = torch.tensor(data=a, dtype=torch.long, device=self.device)
        s_ = torch.tensor(s_, dtype=torch.float, device=self.device)
        r = torch.tensor(r, dtype=torch.float, device=self.device)
        r = torch.unsqueeze(r, dim=-1)
        done = torch.tensor(done, dtype=torch.float, device=self.device)
        done = torch.unsqueeze(done, dim=-1)

        return s, z, a, s_, r, done