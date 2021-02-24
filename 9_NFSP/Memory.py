import torch
import numpy as np
import random


class M_RL:
    # the memory for the reinforcement learning
    # store transition s, a, r, s_
    def __init__(self, capacity, batch_size, device):
        self.capacity = capacity
        self.memory = [0] * self.capacity
        self.batch_size = batch_size
        self.counter = 0
        self.device = device

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        done = 0 if done is False else 1
        self.memory[index] = ([s, a, s_, r, done])
        self.counter += 1

    def get_sample(self):
        batch = random.sample(self.memory[0:min(self.counter, self.capacity)], self.batch_size)
        s, a, s_, r, done = zip(* batch)
        s = torch.tensor(s, dtype=torch.float).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        s_ = torch.tensor(s_, dtype=torch.float).to(self.device)
        r = torch.tensor(r, dtype=torch.float).to(self.device)
        done = torch.tensor(r, dtype=torch.float).to(self.device)
        return s, a, s_, r, done


class M_SL():
    # the memory for the supervised learning
    # store transition s, a, r, s_
    def __init__(self, capacity, batch_size, device):
        self.capacity = capacity
        self.memory = [0] * self.capacity
        self.batch_size = batch_size
        self.counter = 0
        self.device = device

    def store_transition(self, s, a):
        index = self.counter % self.capacity
        self.memory[index] = ([s, a])
        self.counter += 1

    def get_sample(self):
        batch = random.sample(self.memory, self.batch_size)
        s, a = zip(* batch)
        s = torch.tensor(s, dtype=torch.float).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        return s, a
