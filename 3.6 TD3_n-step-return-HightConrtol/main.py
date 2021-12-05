import numpy as np
import matplotlib.pyplot as plt
from Agent import TD3

"""
对于这个任务，初始化的条件很是重要，我们可以看到如果action一直为1
那么position就会无限远的情况，且一直不收敛。
如果前几轮就出现了正向反馈，那么智能体就可以很快地学习到很好的policy
"""
class GYM:
    def __init__(self):
        self.s = None
        self.r = None
        self.done = None

        self.M = 1
        self.G = 9.8
        self.f = 0.1
        self.F_mid = 10
        self.t = 0.1

    def reset(self):
        self.s = None
        self.r = None
        self.done = None

        position = 0
        velocity = 0
        acceleration = 0
        target = np.random.rand()*40. + 30.
        # 被注释掉的是将目前状态与目标一起输入的情况，不稳定。
        # 用差分输入，高度控制就会好很多。
        # self.s = np.array([position, velocity, acceleration, target])
        self.s = np.array([position-target, velocity, acceleration])
        return self.s

    def step(self, a):
        F = 10 * (a+1)
        position = self.s[0]
        velocity = self.s[1]
        acceleration = self.s[2]
        # target = self.s[3]

        f = -self.f * np.sign(velocity) * velocity**2
        F = F + f - self.M*self.G

        acceleration_ = F/self.M
        velocity_ = velocity + self.t*(acceleration + acceleration_)/2
        position_ = position + self.t*(velocity + velocity_)/2

        self.s = np.array([position_, velocity_, acceleration_])
        # 回报函数的设置，设置增量或者绝对差值，都是可以的
        self.r = np.abs(position) - np.abs(position_)
        # self.s = np.array([position_, velocity_, acceleration_, target])
        # self.r = -np.abs(position_ - target)
        self.done = False
        return self.s, self.r, self.done, None


env = GYM()
S_DIM = 3                                       # state dimension
A_DIM = 1                                       # action dimension
BOUND = 1.                                      # bound value of the action
CAPACITY = 10000                                # maximum number of samples stored in memory
BATCH_SIZE = 256                                # the number of samples for each train
HIDDEN = 16                                     # hidden node
LR_ACTOR = 5e-3                                 # learning rate for actor
LR_CRITIC = 5e-3                                # learning rate for critic
ALPHA = 0.3                                     # PRE sampling coefficient
BETA = 1                                        # PRE weight coefficient
VARIANCE_START = 1                              # the initial random factor
REG_COE = 0.000                                 # L2 regularization coefficient
                                                # currently speaking, regularization performs worse
VARIANCE_DECAY = 0.999                          # the decay rate of random factor for each step
VARIANCE_MIN = 0.05                             # the minimum random factor
GAMMA = 0.9                                     # discounting factor
RETURN_STEP = 6                                 # the length of n-step return
TAU = 0.05                                      # soft-update parameters
POLICY_NOISE = 0.2                              # the sigma for policy noise
NOISE_CLIP = 0.5                                # noise clip
POLICY_FREQ = 2                                 # frequency of update policy
MAX_EPISODE = 1000                              # maximum episode to play
MAX_EP_STEPS = 256                              # maximum steps for each episode
RENDER = False                                  # whether render


agent = TD3(s_dim=S_DIM,
            a_dim=A_DIM,
            capacity=CAPACITY,
            batch_size=BATCH_SIZE,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            hidden=HIDDEN,
            reg_coe=REG_COE,
            var_init=VARIANCE_START,
            var_decay=VARIANCE_DECAY,
            var_min=VARIANCE_MIN,
            gamma=GAMMA,
            return_step=RETURN_STEP,
            tau=TAU,
            policy_noise=POLICY_NOISE,
            noise_clip=NOISE_CLIP,
            policy_freq=POLICY_FREQ)

# randomly get some samples
for episode in range(4):
    s = env.reset()
    ep_r = 0
    for ep_step in range(MAX_EP_STEPS):
        a = np.array([np.random.rand()*2-1])
        s_, r, done, info = env.step(a[0])
        agent.store_transition(s, a, s_, r)
        s = s_
    agent.ep_end()

# train part
for episode in range(20):
    s = env.reset()
    ep_r = 0
    for ep_step in range(MAX_EP_STEPS):
        a = agent.get_action(s)
        s_, r, done, info = env.step(a[0]*BOUND)
        agent.store_transition(s, a, s_, r)
        # agent.learn()
        s = s_
        ep_r += r
    agent.ep_end()
    for i in range(MAX_EP_STEPS):
        agent.learn()
    print('Episode:', episode, ' Reward: %i' % int(ep_r), 'var', agent.var)


# Test Part
position = []
velocity = []
acceleration = []
action = []
s = env.reset()
action.append(0)
position.append(s[0])
velocity.append(s[1])
acceleration.append(s[2])
for ep_step in range(MAX_EP_STEPS):
    # interact environment
    a = agent.get_action(s)

    s_, r, done, info = env.step(a[0])
    action.append(a[0])
    position.append(s[0])
    velocity.append(s[1])
    acceleration.append(s[2])

    s = s_

index = np.array(range(len(position)))
plt.plot(index, action, label='action')
plt.plot(index, position, label='position')
plt.plot(index, velocity, label='velocity')
plt.plot(index, acceleration, label='acc')
plt.legend()
plt.show()