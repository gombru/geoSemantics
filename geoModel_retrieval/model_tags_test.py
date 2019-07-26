import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.extra_net = MMNet()

    def forward(self, tag, lat, lon):
        query = self.extra_net(tag, lat, lon)
        return query


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        # loc layers
        self.fc_loc = BasicFC_BN(2, 10)

        # mm tag|loc layers
        self.fc_mm_1 = BasicFC_BN(310, 1024)
        self.fc_mm_2 = BasicFC_BN(1024, 1024)
        self.fc_mm_3 = BasicFC_BN(1024, 1024)
        self.fc_mm_4 = nn.Linear(1024, 300)

    def forward(self, tag, lat, lon):

        # Location
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # MM tag|loc
        anchor = torch.cat((loc, tag), dim=1)
        anchor = self.fc_mm_1(anchor)
        anchor = self.fc_mm_2(anchor)
        anchor = self.fc_mm_3(anchor)
        anchor = self.fc_mm_4(anchor)

        return anchor


class BasicFC_BN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)  # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)