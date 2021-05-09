import random

class ReplayBuffer:
    # the replay buffer used for store and sample transitions.
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = [0] * self.capacity
        self.batch_size = batch_size
        self.counter = 0

    def store_transition(self, s, a, s_, r, done):
        index = self.counter % self.capacity
        done = [1 if done_i else 0 for done_i in done]
        self.memory[index] = [s, a, s_, r, done]
        self.counter += 1

    def get_sample(self):
        batch = random.sample(self.memory[0:min(self.counter, self.capacity)], self.batch_size)
        s, a, s_, r, done = zip(* batch)
        return s, a, s_, r, done