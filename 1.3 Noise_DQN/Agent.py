import torch
import numpy as np
from Network import NoisyNet
from ReplayBuffer import ReplayBuffer


class DQN:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            capacity,
            batch_size,
            lr,
            std_init,
            gamma,
            replace_target_iter
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.std_init = std_init
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter

        # Network
        self.Q = NoisyNet(s_dim, a_num, std_init).to(self.device)
        self.Q_target = NoisyNet(s_dim, a_num, std_init).to(self.device)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # replay buffer, or memory
        self.memory = ReplayBuffer(capacity, batch_size, device)

    def get_action(self, s):
        # epsilon-greedy(Q)
        s = torch.FloatTensor(s).to(self.device)
        q = self.Q(s)
        a = torch.argmax(q).detach().item()
        return a

    def learn(self):
        # samples from memory
        s, a, s_, r, done = self.memory.get_sample()

        # calculate loss function
        index = torch.LongTensor(range(len(r)))
        q = self.Q(s)[index,a]
        q_target = self.Q_target(s_)
        td_target = r + (1-done)*self.gamma * torch.max(q_target, dim=1).values.detach()
        td_error = td_target - q
        loss = torch.mean(torch.pow(td_error,2))

        # train the network
        self.opt.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        self.opt.step()  # apply gradients

        # hard update
        if self.memory.counter%self.replace_target_iter == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())