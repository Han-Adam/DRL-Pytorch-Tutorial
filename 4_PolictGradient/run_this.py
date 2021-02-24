import gym
from RL_Brain import RL_Brain

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

rl_brain = RL_Brain(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.02,
                    reward_decay=0.99,
)

for i_episode in range(500):
    print(i_episode,end='= ')

    s = env.reset()

    while True:
        #env.render()
        a = rl_brain.choose_action(s)
        s_, r, done, info = env.step(a)

        rl_brain.store_transition(s, a, r)
        s = s_

        if done:
            rl_brain.learn()
            break