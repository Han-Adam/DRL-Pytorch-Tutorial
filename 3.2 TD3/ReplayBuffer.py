import numpy as np
import torch
import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = [0] * capacity
        self.batch_size = batch_size
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        done = 0 if done is False else 1
        self.memory[index] = [s, a, s_, r, done]
        self.counter += 1

    def get_sample(self):
        samples_indices = random.choices(range(min(self.capacity, self.counter)), k=self.batch_size)
        samples = np.array(self.memory)[samples_indices]
        s, a, s_, r, done = zip(*samples)

        # samples = np.array(self.memory)[samples_indices]
        # s, a, s_, r, done = zip(* samples)
        s = torch.tensor(s, dtype=torch.float)        # [batch_size, s_dim]
        a = torch.tensor(a, dtype=torch.float)        # [batch_size, a_dim]
        s_ = torch.tensor(s_, dtype=torch.float)      # [batch_size, s_dim]
        r = torch.tensor(r, dtype=torch.float)        # [batch_size]
        r = torch.unsqueeze(r, dim=-1)                # [batch_size, 1]
        done = torch.tensor(done, dtype=torch.float)  # [batch_size]
        done = torch.unsqueeze(done, dim=-1)          # [batch_size, 1]
        return s, a, s_, r, done
