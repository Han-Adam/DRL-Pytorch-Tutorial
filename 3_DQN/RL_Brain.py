import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Deep Q Network, off-policy
class RL_Brain:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            replace_target_iter=300,
            memory_capacity=500,
            batch_size=32,
            greedy_increase = 0.01
    ):
        # Parameter Initialization
        self.n_actions = n_actions
        self.epsilon = 0
        self.replace_target_iter = replace_target_iter
        self.greedy_increase = greedy_increase

        # replay buffer, or memory
        self.memory = Memory(state_dim=n_features,
                             batch_size=batch_size,
                             memory_capacity=memory_capacity)
        # Network
        self.Q = Net(n_input= n_features, n_hidden= 2 * n_actions, n_output= n_actions)
        self.Q_target = Net(n_input= n_features, n_hidden= 2 * n_actions, n_output= n_actions)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr= learning_rate)
        self.Q_target.load_state_dict(self.Q.state_dict())
        # loss function
        self.loss_func = Loss(self.Q, self.Q_target, reward_decay)

    def store_transition(self, s, a, s_, r):
        self.memory.store_transition(s, a, s_, r)

    def choose_action(self, s):
        # epsilon-greedy(Q)
        if np.random.rand() < self.epsilon:
            s = torch.tensor(s, dtype=torch.float)
            actions_value = self.Q(s)
            action = torch.argmax(actions_value)
            action = action.numpy()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        s, a, s_, r = self.memory.get_sample()
        # 训练网络
        loss = self.loss_func.get_loss(s,a,s_,r)
        self.optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        self.optimizer.step()  # apply gradients
        #增大epsilon
        self.epsilon = self.epsilon+self.greedy_increase
        #周期性更新Q_target
        if self.memory.counter%self.replace_target_iter == 0:
            print('replace')
            self.Q_target.load_state_dict(self.Q.state_dict())


# 一个简单的网络，输入是state的维度，输出是one-hot形式
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_input,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self,x):
        hidden = F.relu(self.hidden(x))
        output = self.predict(hidden)
        return output


class Loss:
    def __init__(self, Q, Q_target, gamma):
        self.Q = Q
        self.Q_target = Q_target
        self.gamma = gamma

    def get_loss(self,s,a,s_,r):
        q = self.Q(s)
        q_target = self.Q_target(s_)

        td_target = q.clone().detach()
        # 在pytorch中，做index必须是torch.long
        index = torch.tensor(range(len(r)),dtype=torch.long)
        td_target[index, a] = r + self.gamma * torch.max(q_target,dim=1).values.detach()
        loss = F.mse_loss(q,td_target)
        return loss


# 记忆库
class Memory:
    def __init__(self, state_dim, batch_size, memory_capacity):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory_s = torch.tensor([[0] * state_dim for _ in range(memory_capacity)], dtype=torch.float)
        self.memory_a = torch.tensor([0]*memory_capacity, dtype=torch.long)
        self.memory_r = torch.tensor([0]*memory_capacity, dtype=torch.float)
        self.memory_s_ = torch.tensor([[0] * state_dim for _ in range(memory_capacity)], dtype=torch.float)
        self.counter = 0

    def store_transition(self, s, a, s_, r):
        # 用新的状态转移取代最早出现的转移 replace buffer
        index = self.counter % self.memory_capacity
        self.memory_s[index] = torch.tensor(s, dtype=torch.float)
        self.memory_a[index] = torch.tensor(a, dtype=torch.long)
        self.memory_r[index] = torch.tensor(r, dtype=torch.float)
        self.memory_s_[index] = torch.tensor(s_, dtype=torch.float)
        self.counter += 1

    def get_sample(self):
        sample_index = np.random.choice(
            a=min(self.memory_capacity, self.counter),
            size=self.batch_size)

        s = self.memory_s[sample_index]
        a = self.memory_a[sample_index]
        r = self.memory_r[sample_index]
        s_ = self.memory_s_[sample_index]
        return s, a, s_, r