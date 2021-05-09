import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from Network import Actor, Critic
from ReplayBuffer import ReplayBuffer

class MADDPG:
    def __init__(self,
                 device,
                 agent_num,
                 state_shape_n,
                 action_shape_n,
                 gamma,
                 tau,
                 max_grad_norm,
                 hidden,
                 lr_a,
                 lr_c,
                 buffer_capacity,
                 batch_size
                 ):
        # hyper parameters
        self.device = device
        self.agent_num = agent_num
        self.state_shape_n = state_shape_n
        self.action_shape_n = action_shape_n
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        # define all the actor/critic network
        self.actors = [None for _ in range(self.agent_num)]
        self.critics = [None for _ in range(self.agent_num)]
        self.actors_target = [None for _ in range(self.agent_num)]
        self.critics_target = [None for _ in range(self.agent_num)]
        self.optimizers_a = [None for _ in range(self.agent_num)]
        self.optimizers_c = [None for _ in range(self.agent_num)]
        for i in range(self.agent_num):
            # define actor for the i-th agent
            self.actors[i] = Actor(state_n=state_shape_n[i],
                                   action_n=action_shape_n[i],
                                   hidden=hidden).to(self.device)
            self.actors_target[i] = Actor(state_n=state_shape_n[i],
                                          action_n=action_shape_n[i],
                                          hidden=hidden).to(self.device)
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.optimizers_a[i] = Adam(self.actors[i].parameters(), lr_a)
            # define critic for the i-th agent
            self.critics[i] = Critic(state_n=sum(state_shape_n),
                                     action_n=sum(action_shape_n),
                                     hidden=hidden).to(self.device)
            self.critics_target[i] = Critic(state_n=sum(state_shape_n),
                                     action_n=sum(action_shape_n),
                                     hidden=hidden).to(self.device)
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.optimizers_c[i] = Adam(self.critics[i].parameters(), lr_c)
        # define the memory
        self.batch_size = batch_size
        self.memory = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)

    def get_action(self, s_n):
        # get the action.
        # for "simple_adversary" environment only:
        #       s_n = [[state of 1st agent,length 8],
        #              [state of 2nd agent,length 10],
        #              [state of 3rd agent,length 10]]
        #       a_n = [[action of 1st agent,length 5],
        #              [action of 2nd agent,length 5],
        #              [action of 3rd agent,length 5]]
        a_n = [actor(torch.FloatTensor(s).to(self.device)).detach().cpu().tolist()
               for actor, s in zip(self.actors, s_n)]
        return a_n

    def train(self):
        for index, (actor, actor_target, critic, critic_target, opt_a, opt_c) in \
                enumerate(zip(self.actors,
                              self.actors_target,
                              self.critics,
                              self.critics_target,
                              self.optimizers_a,
                              self.optimizers_c)):
            # get samples from memory
            s_n, a_n, s_n_, r_n, done_n = self.memory.get_sample()
            # get the state/reward/done for index's agent
            s = torch.FloatTensor([s[index] for s in s_n]).to(self.device)             # shape= [batch_size, state_shape_n[index]]
            r = torch.FloatTensor([r[index] for r in r_n]).to(self.device)             # shape= [batch_size]
            done = torch.FloatTensor([done[index] for done in done_n]).to(self.device) # shape= [batch_size]
            # get the concatenated state/action of all agents
            s_n_current = torch.FloatTensor([np.concatenate(s) for s in s_n]).to(self.device)  # shape = [batch_size, sum(state_shape_n)]
            a_n_current = torch.FloatTensor([np.concatenate(a) for a in a_n]).to(self.device)  # shape = [batch_size, sum(action_shape_n)]
            s_n_target = torch.FloatTensor([np.concatenate(s_) for s_ in s_n_]).to(self.device)# shape = [batch_size, sum(state_shape_n)]
            a_n_target = torch.cat(
                [self.actors_target[i](torch.FloatTensor([s[i] for s in s_n]).to(self.device)) # shape = [batch_size, sum(action_shape_n)]
                 for i in range(self.agent_num)], dim=-1)
            # calculate time-difference target
            q_ = critic_target(s_n_target, a_n_target).squeeze()
            td_target = r + (1-done)*self.gamma*q_
            # update critic network
            q = critic(s_n_current, a_n_current).squeeze()
            critic_loss = F.mse_loss(q, td_target)
            opt_c.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            opt_c.step()
            # update actor network
            model_out, policy = actor(s, model_original_out=True)
            a_n_current = torch.FloatTensor(a_n).to(self.device)
            a_n_current[:,index,:] = policy
            a_n_current = torch.reshape(a_n_current,[self.batch_size,-1])
            loss_a = -torch.mean(critic(s_n_current, a_n_current))
            loss_pse = torch.mean(torch.pow(model_out, 2))
            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            opt_a.step()
            # update target network
            self.soft_update(actor_target, actor)
            self.soft_update(critic_target, critic)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
