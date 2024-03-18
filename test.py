import torch
import torch.nn as nn

input_dim, hidden_dim, num_layers = 2, 2, 5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, observations, hidden=None, return_hidden=True):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary

net = Net()

observation1 = torch.zeros(size=[4, input_dim], dtype=torch.float)
hn_0 = torch.zeros(size=[num_layers, input_dim], dtype=torch.float)
cn_0 = torch.zeros(size=[num_layers, input_dim], dtype=torch.float)

summary_, (hn_, cn_) = net(observation1, (hn_0, cn_0))

print('no batch')
print(observation1.shape)
print(summary_.shape)
print(hn_.shape)
print(cn_.shape)

print('batch')


observation2 = torch.zeros(size=[4, input_dim], dtype=torch.float)

observation3 = torch.ones(size=[4, input_dim], dtype=torch.float)

observation = torch.zeros(size=[3, 4, input_dim])
observation[0] = observation1
observation[1] = observation2
observation[2] = observation3
print(observation.shape)
hn_0 = torch.zeros(size=[num_layers, 3, 2])
cn_0 = torch.zeros(size=[num_layers, 3, 2])

summary_, (hn_, cn_) = net(observation, (hn_0, cn_0))

print(summary_.shape)
print(hn_.shape)
print(cn_.shape)


x = [0] * 10
x[1] += 1
x[1] += 1
x[5] += 1
print(x)

print('\n\n\n\n testing gather')
x = torch.zeros(size=[32, 4])
for i in range(32):
    for j in range(4):
            x[i, j] = 100 * i + 10 * j

index = torch.LongTensor(range(32))
# index = torch.LongTensor([1, 2, 3, 4, 5])
a = torch.randint(low=0, high=4, size=[32])
x = x[index, a]
print(x.shape)

print('\n\n\n\n testing gather, with batch and trajectory length')
batch_size = 2
x = torch.zeros(size=[batch_size, 4, 5])
for i in range(batch_size):
    for j in range(4):
        for k in range(5):
            x[i, j, k] = 100 * i + 10 * j + k
print(x)

index = torch.LongTensor(range(batch_size))
index = torch.vstack([index] * 4).T
# index = torch.reshape(torch.concat([index] * 4, dim=-1), shape=[32, 4])
print(index.shape)
print(index)
index2 = torch.torch.LongTensor(range(4))
index2 = torch.vstack([index2] * 2)
print(index2.shape)
print(index2)

# index = torch.LongTensor([1, 2, 3, 4, 5])
a = torch.randint(low=0, high=5, size=[batch_size, 4])
x = x[index, index2, a]
print(x.shape)
print(x)
