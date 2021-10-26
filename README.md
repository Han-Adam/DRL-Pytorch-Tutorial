# DRL-Pytorch-Tutorial

`This` is a tutorial for deep reinforcement learning (DRL). This tutorial mainly aims at the beginner of DRL. The code is in its simplest version. We mainly focus on the environment of 'CartPole-v0' and 'Pendulum-v0' in [OpenAI-Gym](https://github.com/openai/gym), which could be viewed as MNIST data set in computer vision task. The environment is very simple. Therefore, you could observe the convergence of the code within several minutes (even shorter), even the hyperparameters are not specifically selected. After all, you could use this to quickly learn the principle of several DRL algorithm, but you'd better not use this code for serious experiment.  

## Requirement

Most of the experiment depends on Python 3.7 expect MADDPG.

## Main Structure

`main.py`: this script contains the setting of hyperparameters and main loop, always need to import `Agent`.

`Agent.py`: contain the action selection function, learning algorithm, always need to import `Network` and `ReplayBuffer`.

`Network.py`: contain the network that need to use in this algorithm.

`ReplayBuffer.py`: contain the memory.

## Paper List

1.1 DQN

The beginning of the DRL. you cannot skip algorithm this if you want to play with DRL. Several point need to mentioned for this algorithm: (1) we use a Q-value function to approximate the "discounted reward" (also refer to "return" in some literature); (2) we need an additional target network to stabilize the train process; (3) a replay buffer is utilized to reuse previous experience. 

[[1312.5602\] Playing Atari with Deep Reinforcement Learning (arxiv.org)](https://arxiv.org/abs/1312.5602)

[Human-level control through deep reinforcement learning (nature.com)](https://www.nature.com/articles/nature14236.pdf)

Also, there are several improved version of DQN. All of these techniques are integrated in 'Rainbow'. These techniques are not implemented in this tutorial. 

[[1710.02298\] Rainbow: Combining Improvements in Deep Reinforcement Learning (arxiv.org)](https://arxiv.org/abs/1710.02298)

1.2 Prioritized Experience Replay

In normal replay buffer, every single transition has same probability to be used to train our model. This paper give different transition with different priority, which makes models are more likely to learn difficult transitions. This replay buffer could be embedded into various DRL architecture, while we use DQN to give a simple demonstration. 

[[1511.05952\] Prioritized Experience Replay (arxiv.org)](https://arxiv.org/abs/1511.05952)

1.3 Noise Networks

For normal DQN, we use epsilon-greedy for exploration. This algorithm introduce uncertainty into the network itself, which makes the exploration more intelligent.

[[1706.10295\] Noisy Networks for Exploration (arxiv.org)](https://arxiv.org/abs/1706.10295)

2.1-2.2 Policy Gradient & Actor Critic

These two algorithm are too old, I cannot find a original paper that proposed this method. But mostly, researchers will refer to this:

[CiteSeerX — Policy Gradient Methods for Reinforcement Learning with Function Approximation (psu.edu)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.6.696)

For Q-learning based method, we always want find a value function (also refer to critic) and determine our action based on "expected return". But this time, we train a policy (also refer to actor) to directly determine the action. This method also introduce randomness into the action selection process, while the value function-based algorithm is always deterministic.

2.3-2.4 Asynchronous Advantage Actor Critic (A2C & A3C)

This algorithm introduces distributed computing into DRL. When we refer to A2C, it means Advantage Actor Critic. Please refer [MorvanZhou/pytorch-A3C](https://github.com/MorvanZhou/pytorch-A3C) for the original code, I did not implement the algorithm myself.

[[1602.01783v2\] Asynchronous Methods for Deep Reinforcement Learning (arxiv.org)](https://arxiv.org/abs/1602.01783v2)

2.5  Generalized Advantage Estimation (GAE)

This algorithm generalized the advantage formula. In this artical, the TD-error (one step of advantage) and MT-error (discounted reward error) is unified. It is widely used in the later algorithms.

[[1506.02438v4\] High-Dimensional Continuous Control Using Generalized Advantage Estimation (arxiv.org)](https://arxiv.org/abs/1506.02438v4)

3 DDPG

Recall all the algorithms before, we always have a discrete action space. However, in reality, the action space is always continuous. from now on, we will focus on continuous action space problem.

For policy gradient in 2.1, the network will output a probability distribution among a discrete action space. For continuous action space, the network could output a set of parameters, which determine a probability distribution (eg. Gaussian distribution). then, we could select an action according to the parameterized distribution. This algorithm accords with policy gradient, but not converge in practice.

Now, let the variance of the distribution becomes zero. Then, the policy gradient changes from stochastic version to deterministic version. For a detailed mathematical explanation of DPG, please refer to:

[Deterministic Policy Gradient Algorithms](https://deepmind.com/research/publications/deterministic-policy-gradient-algorithms)

If we use deep neural network to approximate the policy and value function, then, we get DDPG:

[[1509.02971v2\] Continuous control with deep reinforcement learning (arxiv.org)](https://arxiv.org/abs/1509.02971v2)

Also, please refer to this paper for TD3, an improved version of DDPG:

[[1802.09477v3\] Addressing Function Approximation Error in Actor-Critic Methods (arxiv.org)](https://arxiv.org/abs/1802.09477v3)

4 PPO

This algorithm is totally different from the family of policy gradient. Firstly, please refer to TRPO for a detailed (complex) mathematical description:

[[1502.05477v5\] Trust Region Policy Optimization (arxiv.org)](https://arxiv.org/abs/1502.05477v5)

There are two version of PPO, from DeepMind and OpenAI respectively:

[[1707.02286v2\] Emergence of Locomotion Behaviours in Rich Environments (arxiv.org)](https://arxiv.org/abs/1707.02286v2)

[[1707.06347\] Proximal Policy Optimization Algorithms (arxiv.org)](https://arxiv.org/abs/1707.06347)

In this tutorial, we mainly focus on the algorithm in the second paper (OpenAI version). Moreover, thy have two version of PPO: Clipping & Adaptive KL Penalty. In this tutorial, we focus on the clipping version, which performed better according to the paper.

5 MADDPG

for the previous work, we always focus on the control of one single agent. This algorithm generalize the DDPG into the multi-agent environment. 

[[1706.02275v4] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (arxiv.org)](https://arxiv.org/abs/1706.02275v4)

To implement this algorithm, please follow the instruction in [OpenAI/MADDPG](https://github.com/openai/maddpg) to use a specific version of gym. Python 3.5 and Python 3.7 are both available, while the you need a right version of gym.

Rum `main.py` for training process. It will also store your trained agent and reward. Run `plot.py` to view the reward. Run `evaluate.py` to generate a video of multi agent game. the `ENV_NAME` could be `simple_push` or `simple_adversary.`

6.1-6.2 SAC

Soft Actor Critic, an improved version of actor critic architecture. Recall 3 DDPG, we talk a version of continuous control based on paramertized probability distribution. This is basic frame of SAC. Moreover, it also introduced the entropy into the objective function and proposed soft value function.

There are two versions of SAC:

[[1801.01290\] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (arxiv.org)](https://arxiv.org/abs/1801.01290)

[[1812.05905\] Soft Actor-Critic Algorithms and Applications (arxiv.org)](https://arxiv.org/abs/1812.05905)

Compared with first version, the second one use Q-value function only (first version use both V-value function and Q-value function). It also introduce an "Automating Entropy Adjustment" algorithm.

6.3 DIAYN

Diversity is all you need, what a fashionable title for academic paper. The parameter update algorithm is based on SAC, but the core of this algorithm is its design of reward. No predefined reward function is needed, while this algorithm  uses diversity to define the reward. By running this algorithm, you could observe four distinct paths traveled by the agent, given different skill.

Need a little bit of knowledge of "Information Theory" to understand the proof of this paper.

[[1802.06070v6\] Diversity is All You Need: Learning Skills without a Reward Function (arxiv.org)](https://arxiv.org/abs/1802.06070v6)
