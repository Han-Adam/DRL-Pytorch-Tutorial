import numpy as np
import torch as torch
import Network
import Memory

class Player:
    def __init__(self,
                 index,
                 s_dim,
                 a_dim,
                 rl_capacity,
                 sl_capacity,
                 rl_batch_size,
                 sl_batch_size,
                 rl_lr,
                 sl_lr,
                 gamma,
                 epsilon,
                 tau,
                 device,
                 ):
        self.index = index
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.rl_batch_size = rl_batch_size
        self.sl_batch_size = sl_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.device = device
        # Memory for reinforcement learning
        self.m_rl = Memory.M_RL(capacity=rl_capacity,
                                batch_size=rl_batch_size,
                                device=self.device)
        # Memory for supervise learning
        self.m_sl = Memory.M_SL(capacity=sl_capacity,
                                batch_size=sl_batch_size,
                                device=self.device)
        # Q-Network for reinforcement learning
        self.Q = Network.Q_Net(s_dim=self.s_dim,
                               a_dim=self.a_dim).to(self.device)
        self.Q_target = Network.Q_Net(s_dim=self.s_dim,
                                      a_dim=self.a_dim).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=rl_lr)
        # Pi-Network for supervised learning
        self.Pi = Network.Pi_Net(s_dim=self.s_dim,
                                 a_dim=self.a_dim).to(self.device)
        self.Pi_optimizer = torch.optim.Adam(self.Pi.parameters(), lr=sl_lr)

    def Q_action(self, s):
        # epsilon-greedy(Q)
        if np.random.rand() < self.epsilon:
            s = torch.tensor(s, dtype=torch.float).to(self.device).unsqueeze(0)
            action_value = self.Q(s).squeeze()
            action = torch.argmax(action_value)
            action = action.numpy()
        else:
            action = np.random.randint(0, self.a_dim)
        return int(action)

    def Pi_action(self, s):
        # action across to probability distribution
        s = torch.tensor(s, dtype=torch.float).to(self.device).unsqueeze(0)
        prob = self.Pi(s).squeeze()
        action = np.random.choice(range(self.a_dim), p=prob.detach().numpy())
        return int(action)

    def Q_loss(self, s, a, s_, r, done):
        q = self.Q(s)
        q_target = self.Q_target(s_)
        td_target = q.clone().detach()
        index = torch.tensor(range(self.rl_batch_size), dtype=torch.long)
        td_target[index, a] = r + self.gamma * (1 - done) * torch.max(q_target, dim=1).values.detach()
        loss = torch.mean(torch.pow((td_target - q), 2))
        return loss

    def Q_learn(self):
        # learning process of the Q network
        s, a, s_, r, done = self.m_rl.get_sample()
        loss = self.Q_loss(s, a, s_, r, done)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        self.soft_update(self.Q_target, self.Q)

    def Pi_loss(self, s, a):
        index = torch.tensor(range(len(a)), dtype=torch.long)
        pi = self.Pi(s)
        loss = -torch.mean(torch.log(pi[index, a]))
        return loss

    def Pi_learn(self):
        # learning process of the Pi network
        s, a = self.m_sl.get_sample()
        loss = self.Pi_loss(s, a)
        self.Pi_optimizer.zero_grad()
        loss.backward()
        self.Pi_optimizer.step()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.tau) + param.data * self.tau
            )


class RL_Brain:
    def __init__(self,
                 s_dim,
                 a_dim,
                 rl_capacity,
                 sl_capacity,
                 rl_batch_size,
                 sl_batch_size,
                 rl_lr,
                 sl_lr,
                 gamma,
                 epsilon,
                 tau,
                 device,
                 eta
                 ):
        # definition of two players
        self.player1 = Player(index=1,
                              s_dim=s_dim,
                              a_dim=a_dim,
                              rl_capacity=rl_capacity,
                              sl_capacity=sl_capacity,
                              rl_batch_size=rl_batch_size,
                              sl_batch_size=sl_batch_size,
                              rl_lr=rl_lr,
                              sl_lr=sl_lr,
                              gamma=gamma,
                              epsilon=epsilon,
                              tau=tau,
                              device=device)
        self.player2 = Player(index=2,
                              s_dim=s_dim,
                              a_dim=a_dim,
                              rl_capacity=rl_capacity,
                              sl_capacity=sl_capacity,
                              rl_batch_size=rl_batch_size,
                              sl_batch_size=sl_batch_size,
                              rl_lr=rl_lr,
                              sl_lr=sl_lr,
                              gamma=gamma,
                              epsilon=epsilon,
                              tau=tau,
                              device=device)
        self.rl_batch_size = rl_batch_size
        self.sl_batch_size = sl_batch_size
        self.eta = eta
        self.mode = ['reinforcement', 'supervised']
        self.current_mode = 'reinforcement'

    def select_mode(self):
        # use Q-Net or Pi-Net
        self.current_mode = self.mode[0 if np.random.rand() < self.eta else 1]

    def store_transition(self, s, a, s_, r, done):
        # Store the transition
        self.player1.m_rl.store_transition(s['1'], a['1'], s_['1'], r['1'], done)
        self.player2.m_rl.store_transition(s['2'], a['2'], s_['2'], r['2'], done)
        if self.mode == 'reinforcement':
            self.player1.m_sl.store_transition(s['1'], a['1'])
            self.player2.m_sl.store_transition(s['2'], a['2'])

    def get_action(self, s):
        # get an action
        if self.current_mode == 'reinforcement':
            action1 = self.player1.Q_action(s['1'])
            action2 = self.player2.Q_action(s['2'])
        else:
            action1 = self.player1.Pi_action(s['1'])
            action2 = self.player2.Pi_action(s['2'])
        return action1, action2

    def learn(self):
        # update Network parameters
        if self.player1.m_rl.counter > self.rl_batch_size:
            self.player1.Q_learn()
            self.player2.Q_learn()
        if self.player2.m_sl.counter > self.sl_batch_size:
            self.player1.Pi_learn()
            self.player2.Pi_learn()

