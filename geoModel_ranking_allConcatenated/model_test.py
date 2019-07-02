import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.extra_net = MMNet(self.c)

    def forward(self, img, tag, lat, lon):
        # Here tag is [1x100kx300]
        # Others are [1xk]
        print(img.shape)
        print(tag.shape)
        score = self.extra_net(img, tag, lat, lon)
        return score


class MMNet(nn.Module):

    def __init__(self, c):
        super(MMNet, self).__init__()

        self.fc1 = BasicFC(602, 512)
        self.fc2 = BasicFC(512, 512)
        self.fc3 = BasicFC(512, 512)
        self.fc4 = nn.Linear(512, 1)


    def forward(self, img, tag, lat, lon):

        # Concatenate
        x = torch.cat((img, tag), dim=1)
        x = torch.cat((lat, x), dim=1)
        x = torch.cat((lon, x), dim=1)

        # MLP
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

class BasicFC(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)