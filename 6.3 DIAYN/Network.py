import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, s_dim, z_num, hidden, a_num):
        super(Policy,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_num, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_num)
        )

    def forward(self, s, z, log=False):
        feature = self.net(torch.cat([s,z],dim=-1))
        if log:
            return F.log_softmax(feature, dim=-1)
        else:
            return F.softmax(feature, dim=-1)


class VNet(nn.Module):
    def __init__(self, s_dim, z_num, hidden):
        super(VNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_num, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, z):
        return self.net(torch.cat([s, z], dim=-1))


class QNet(nn.Module):
    def __init__(self, s_dim, z_num, hidden, a_num):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_num, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_num),
        )

    def forward(self, s, z):
        return self.net(torch.cat([s, z], dim=-1))


class Discriminator(nn.Module):
    def __init__(self, s_dim, z_num, hidden):
        super(Discriminator,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_num)
        )

    def forward(self, s, log=False):
        feature = self.net(s)
        if log:
            return F.log_softmax(feature, dim=-1)
        else:
            return F.softmax(feature, dim=-1)



if __name__ == "__main__":
    from torch.distributions import Categorical
    from torch.distributions import OneHotCategorical
    onehot = OneHotCategorical(torch.ones(4))
    s = torch.FloatTensor([1,2])
    z = onehot.sample() #torch.LongTensor([0,0,1,0])
    print(s, z)
    policy = Policy(s_dim=2, z_num=4, hidden=32, a_num=4)
    vnet = VNet(s_dim=2, z_num=4, hidden=32)
    qnet = QNet(s_dim=2, z_num=4, hidden=32, a_num=4)
    dis = Discriminator(s_dim=2, z_num=4, hidden=32)

    prob = policy(s, z)
    print(prob)
    dist = Categorical(prob)
    a = dist.sample()
    print(a)
    index = torch.LongTensor(range(1))
    v = vnet(s, z)
    q = qnet(s, z)
    print(v, q)
    print(index, q[a])  # q[a].unsqueeze(dim=-1))
    prob = dis(s)
    z = z.argmax(dim=-1)
    print(prob)
    print(z)
    print(prob[z])#.unsuqeeze(dim=-1))
