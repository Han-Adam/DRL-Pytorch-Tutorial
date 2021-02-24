import numpy as np
import gym
import lasertag
from wrapper import env_wrap
from RL_Brain import RL_Brain
import torch


if __name__ == '__main__':
    env = gym.make('LaserTag-small2-v0')
    env = env_wrap(env)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    print('brain start building')
    brain = RL_Brain(s_dim=env.observation_space.shape,
                     a_dim=env.action_space.n,
                     rl_capacity=100000,
                     sl_capacity=50000,
                     rl_batch_size=2,
                     sl_batch_size=2,
                     rl_lr=1e-4,
                     sl_lr=1e-4,
                     gamma=0.9,
                     epsilon=0.9,
                     tau=0.1,
                     eta=0.9,
                     device=device
    )
    MAX_EPISODE = 1000
    RENDER = True
    count = 0
    for episode in range(MAX_EPISODE):
        print(episode)
        p1_r_total = 0
        p2_r_total = 0
        p1_s, p2_s = env.reset()
        while True:
            if RENDER:
                env.render()
            count += 1
            p1_a, p2_a = brain.get_action({'1': p1_s, '2': p2_s})
            (p1_s_, p2_s_), r, done, infor = env.step({'1': p1_a, '2': p2_a})
            p1_r, p2_r = r[0], r[1]
            p1_r_total += p1_r
            p2_r_total += p2_r
            brain.store_transition(s={'1': p1_s, '2': p2_s},
                                   a={'1': p1_a, '2': p2_a},
                                   s_={'1': p1_s_, '2': p2_s_},
                                   r={'1': p1_r, '2': p2_r},
                                   done=done)
            brain.learn()
            if done:
                brain.select_mode()
                print('episode:'+str(episode)+' player1:'+str(p1_r_total)+' player2:'+str(p2_r_total))
                break
