import torch
import numpy as np
import torch.nn.functional as F
from Network import DRQ_Net
from ReplayBuffer import ReplayBuffer


class DRQN:
    def __init__(
            self,
            s_dim,
            a_num,
            hidden,
            rnn_num_layers,
            capacity,
            max_trajectory_len,
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
        self.hidden = hidden
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.greedy_increase = greedy_increase
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter

        # Network
        self.Q = DRQ_Net(s_dim, hidden, a_num, num_layers=rnn_num_layers)
        self.Q_target = DRQ_Net(s_dim, hidden, a_num, num_layers=rnn_num_layers)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.hn = torch.zeros(size=[rnn_num_layers, hidden], dtype=torch.float)
        self.cn = torch.zeros(size=[rnn_num_layers, hidden], dtype=torch.float)
        # No need for that, directly set None during training
        # self.hn_train = torch.zeros(size=[rnn_num_layers, batch_size, hidden], dtype=torch.float)
        # self.cn_train = torch.zeros(size=[rnn_num_layers, batch_size, hidden], dtype=torch.float)

        # experience replay buffer, memory
        self.memory = ReplayBuffer(s_dim, capacity, max_trajectory_len, batch_size)
        self.counter = 0

    def get_action(self, s):
        # epsilon-greedy(Q)
        s = torch.FloatTensor(s)
        s = torch.unsqueeze(s, dim=0)
        actions_value, self.hn, self.cn = self.Q(s, self.hn, self.cn, True)
        if np.random.rand() < self.epsilon:
            action = torch.argmax(actions_value)
            action = action.item()
        else:
            action = np.random.randint(0, self.a_num)
        return action

    def learn(self):
        # samples from memory
        s, a, s_, r, done = self.memory.get_sample()
        batch_size, trajectory_length = a.shape
        # calculate loss function
        index1 = torch.LongTensor(range(batch_size))
        index1 = torch.vstack([index1] * trajectory_length).T
        index2 = torch.LongTensor(range(trajectory_length))
        index2 = torch.vstack([index2] * batch_size)
        q = self.Q(s)[index1, index2, a]

        with torch.no_grad():
            q_target = self.Q_target(s_)
            td_target = r + (1-done) * self.gamma * torch.max(q_target, dim=2).values
        loss = F.mse_loss(q, td_target)
        # train the network
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # renew epsilon
        self.epsilon = min(self.epsilon + self.greedy_increase, 1)
        # hard update
        self.counter += 1
        if self.counter % self.replace_target_iter == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
