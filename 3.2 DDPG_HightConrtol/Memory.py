import numpy as np
import torch
import random


class Memory:
    def __init__(self, capacity, batch_size, device):
        self.capacity = int(capacity)
        self.memory = [0] * self.capacity
        self.batch_size = batch_size
        self.device = device
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        s, a, s_ = s.tolist(), a.tolist(), s_.tolist()
        done = 1 if done else 0
        self.memory[index] = [s, a, s_, r, done]
        self.counter += 1

    def get_sample(self):
        samples_indices = random.choices(range(min(self.capacity, self.counter)), k=self.batch_size)
        samples = np.array(self.memory)[samples_indices]
        s, a, s_, r, done = zip(*samples)

        s = torch.tensor(s,
                         dtype=torch.float,
                         device=self.device)     # shape = [batch_size, s_dim]
        a = torch.tensor(a,
                         dtype=torch.float,
                         device=self.device)     # shape = [batch_size, a_dim]
        s_ = torch.tensor(s_,
                          dtype=torch.float,
                          device=self.device)    # shape = [batch_size. s_dim]
        r = torch.tensor(r,
                         dtype=torch.float,
                         device=self.device)     # shape = [batch_size]
        r = r.unsqueeze(dim = -1)                # shape = [batch_size, 1]
        done = torch.tensor(done,
                            dtype=torch.float,
                            device=self.device)  # shape = [batch_size]
        done = done.unsqueeze(dim = -1)          # shape = [batch_size, 1]

        return s, a, s_, r, done