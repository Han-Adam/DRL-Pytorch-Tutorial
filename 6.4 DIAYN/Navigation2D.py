import matplotlib.pyplot as plt
import numpy as np
from gym.spaces.discrete import Discrete

class Navigation2D :

    def __init__(self, _iter):
        self.reset()
        self.observation_space = np.array([0,0])
        self.iter = _iter
        self.action_space = Discrete(4)
        self.fig_num = plt.figure().number
        plt.close()

    def step(self,a):

        if a == 0 :
            self.x += 0.1
            self.y += 0.1
        elif a == 1 :
            self.x += 0.1
            self.y += -0.1
        elif a == 2 :
            self.x += -0.1
            self.y += 0.1
        elif a == 3 :
            self.x += -0.1
            self.y += -0.1
        else :
            print('ERROR')
        if self.x <-1 :
            self.x = -1
        if self.x > 1:
            self.x = 1

        if self.y < -1:
            self.y = -1
        if self.y > 1:
            self.y = 1

        self.memory.append(np.array([self.x, self.y]))

        if len(self.memory)>self.iter :
            done = True
        else :
            done = False
        return np.array([self.x, self.y]), False, done, False

    def render(self):
        plt.figure(self.fig_num)
        plt.clf()
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.plot(*np.stack(self.memory, axis=1)) # History
        plt.plot([self.memory[-1][0]], [self.memory[-1][1]], "o") # Current position
        plt.pause(1/60) # 60 frames per second

    def plot_res(self):
        plt.figure()
        plt.ylim(-0.1, 1.1)
        plt.xlim(-0.1, 1.1)
        plt.plot(plt.plot([i for i,j in self.memory ],[j for i,j in self.memory ]))
        plt.show(block = False)

    def reset(self):
        self.x = 0
        self.y = 0
        self.memory = [np.array([self.x, self.y])]
        return(np.array([self.x, self.y]))

    def close(self):
        plt.figure(self.fig_num)
        plt.close()