import torch
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, s_dim, capacity, max_trajectory_len, batch_size):
        self.capacity = capacity
        self.max_trajectory_len = max_trajectory_len
        self.s = torch.zeros(size=[capacity, max_trajectory_len, s_dim], dtype=torch.float, requires_grad=False)
        self.a = torch.zeros(size=[capacity, max_trajectory_len], dtype=torch.long, requires_grad=False)
        self.s_ = torch.zeros(size=[capacity, max_trajectory_len, s_dim], dtype=torch.float, requires_grad=False)
        self.r = torch.zeros(size=[capacity, max_trajectory_len], dtype=torch.float, requires_grad=False)
        self.done = torch.zeros(size=[capacity, max_trajectory_len], dtype=torch.float, requires_grad=False)
        self.trajectory_len = np.zeros(shape=[capacity], dtype=np.int32)
        self.ready_for_sample = np.zeros(shape=[capacity], dtype=np.int32)
        self.batch_size = batch_size
        self.new_trajectory = True
        self.total_counter = 0
        self.step_counter = 0

    def store_transition(self, s, a, s_, r, done):
        trajectory_counter = self.total_counter % self.capacity
        # starting new trajectory, empty the previous memory
        if self.new_trajectory:
            self.s[trajectory_counter] = 0
            self.a[trajectory_counter] = 0
            self.s_[trajectory_counter] = 0
            self.r[trajectory_counter] = 0
            self.done[trajectory_counter] = 0
            self.trajectory_len[trajectory_counter] = 0
            self.ready_for_sample[trajectory_counter] = 0
            self.step_counter = 0
            self.new_trajectory = False

        # stor transition
        self.s[trajectory_counter, self.step_counter] = torch.tensor(s, dtype=torch.float)
        self.a[trajectory_counter, self.step_counter] = torch.tensor(a, dtype=torch.long)
        self.s_[trajectory_counter, self.step_counter] = torch.tensor(s_, dtype=torch.float)
        self.r[trajectory_counter, self.step_counter] = torch.tensor(r, dtype=torch.float)
        self.done[trajectory_counter, self.step_counter] = torch.tensor(1 if done else 0, dtype=torch.long)
        self.trajectory_len[trajectory_counter] += 1
        # if the current trajectory is finish, or the memory is full, start another trajectory storage
        if done or self.trajectory_len[trajectory_counter] == self.max_trajectory_len:
            self.ready_for_sample[trajectory_counter] = 1
            self.total_counter += 1
            self.new_trajectory = True

        self.step_counter += 1

    def get_sample(self):
        options = np.where(self.ready_for_sample == 1)[0]
        choices = np.random.choice(options, size=self.batch_size)
        length_of_choices = self.trajectory_len[choices]
        available_length = np.min(length_of_choices)
        s = self.s[choices, 0:available_length]              # [batch, length, s_dim]
        a = self.a[choices, 0:available_length]              # [batch, length]
        s_ = self.s_[choices, 0:available_length]            # [batch, length, s_dim]
        r = self.r[choices, 0:available_length]              # [batch, length]
        done = self.done[choices, 0:available_length]        # [batch, length]
        return s, a, s_, r, done