import torch
import numpy as np
import torch.nn.functional as F
from Network import Q_Net
from ReplayBuffer import ReplayBuffer


class DQN:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            hidden,
            capacity,
            batch_size,
            lr,
            epsilon_start,
            greedy_increase,
            gamma,
            replace_target_iter
    ):
        # Parameter Initialization
        self.s_dim = s_dim
        self.a_num = a_num
        self.device = device
        self.hidden = hidden
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.greedy_increase = greedy_increase
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter

        # Network
        self.Q = Q_Net(s_dim, hidden, a_num).to(self.device)
        self.Q_target = Q_Net(s_dim, hidden, a_num).to(self.device)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # experience replay buffer, memory
        self.memory = ReplayBuffer(s_dim, capacity, batch_size)

    def get_action(self, s):
        # epsilon-greedy(Q)
        if np.random.rand() < self.epsilon:
            s = torch.FloatTensor(s).to(self.device)
            actions_value = self.Q(s)
            action = torch.argmax(actions_value)
            action = action.item()
        else:
            action = np.random.randint(0, self.a_num)
        return action

    def learn(self):
        # samples from memory
        s, a, s_, r, done = self.memory.get_sample()
        # calculate loss function
        index = torch.LongTensor(range(len(r)))
        q = self.Q(s)[index, a]
        with torch.no_grad():
            q_target = self.Q_target(s_)
            td_target = r + (1-done) * self.gamma * torch.max(q_target, dim=1).values
        loss = F.mse_loss(q, td_target)
        # train the network
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # renew epsilon
        self.epsilon = min(self.epsilon + self.greedy_increase, 1)
        # hard update
        if self.memory.counter % self.replace_target_iter == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
