import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.extra_net = MMNet()

    def forward(self, img):

        img = self.extra_net(img)
        return img


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        # img layers
        self.fc_i_1 = BasicFC_BN(300, 1024)
        self.fc_i_2 = nn.Linear(1024, 1024)
        self.fc_i_3 = nn.Linear(1024, 1024)

    def forward(self, img):
        # Images embeddings processing
        img = self.fc_i_1(img)
        img = self.fc_i_2(img)
        img = self.fc_i_3(img)

        return img


class BasicFC_BN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)  # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)