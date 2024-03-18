import torch
import numpy as np
import torch.nn.functional as F
from Network import Q_Net
from ReplayBuffer import PrioritizedReplayBuffer


class DQN:
    def __init__(
            self,
            s_dim,
            a_num,
            device,
            hidden,
            capacity,
            batch_size,
            rank,
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
        self.rank = rank
        self.epsilon = epsilon_start
        self.greedy_increase = greedy_increase
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter

        # Network
        self.Q = Q_Net(s_dim, hidden, a_num).to(self.device)
        self.Q_target = Q_Net(s_dim, hidden, a_num).to(self.device)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # replay buffer, or memory
        self.memory = PrioritizedReplayBuffer(s_dim, capacity, batch_size, rank)

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
        s, a, s_, r, done, weight, samples_index = self.memory.get_sample()
        # calculate loss function
        index = torch.LongTensor(range(len(r)))
        q = self.Q(s)[index, a]
        with torch.no_grad():
            q_target = self.Q_target(s_)
            td_target = r + (1-done) * self.gamma * torch.max(q_target, dim=1).values
            td_error = td_target - q
        loss = torch.mean(weight * (q - td_target) ** 2)
        # loss = F.mse_loss(q, td_target)
        # train the network
        self.opt.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        self.opt.step()  # apply gradients
        # renew epsilon
        self.epsilon = min(self.epsilon + self.greedy_increase, 1)
        # renew the priority of memory
        new_priority = torch.abs(td_error).numpy() + (np.e ** -10)  # + (np.e ** -10))**self.memory.alpha
        self.memory.priority[samples_index] = new_priority
        # hard update
        if self.memory.counter%self.replace_target_iter == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())