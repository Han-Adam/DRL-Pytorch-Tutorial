import gym
import numpy as np
from RL_Brain import RL_Brain

RENDER = True

# 主程序中，所有的数据都以numpy的形式运行
# 当数据传输到brain中时，我们再将其转化为torch.tensor的形式。

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space.n)
print(env.observation_space.shape[0])
print(env.observation_space.high)
print(env.observation_space.low)

rl_brain = RL_Brain(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.01,
                    replace_target_iter=100,
                    memory_capacity=2000,)
total_steps = 0

for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        if RENDER:
            env.render()
        # 采取行动
        action = rl_brain.choose_action(observation)
        # 更新环境
        observation_, reward, done, info = env.step(action)
        # 对reward进行加工
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2
        # 存储transition
        rl_brain.store_transition(observation, action, observation_, reward)
        ep_r += reward
        if total_steps > 100:
            # 学习
            rl_brain.learn()

        if done:
            # print('episode: ', i_episode,
            #       'ep_r: ', round(ep_r, 2),
            #       ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1