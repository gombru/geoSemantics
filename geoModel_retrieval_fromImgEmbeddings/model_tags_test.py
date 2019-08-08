import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mm_net = MMNet()

    def forward(self, tag, lat, lon):
        query = self.mm_net(tag, lat, lon)
        return query

class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        # loc layers
        self.fc_loc_1 = BasicFC(2, 300)
        self.fc_loc_2 = BasicFC(300, 300)

        # mm tag|loc layers
        self.fc_mm_1 = BasicFC_GN(600, 1024)
        self.fc_mm_2 = BasicFC_GN(1024, 2048)
        self.fc_mm_3 = BasicFC_GN(2048, 2048)
        self.fc_mm_4 = BasicFC_GN(2048, 1024)
        self.fc_mm_5 = nn.Linear(1024, 300)

    def forward(self, tag, lat, lon):

        # Location
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc_1(loc)
        loc = self.fc_loc_2(loc)
        loc_norm = loc.norm(p=2, dim=1, keepdim=True)
        loc = loc.div(loc_norm)
        loc[loc != loc] = 0  # avoid nans

        # MM tag|loc
        anchor = torch.cat((loc, tag), dim=1)
        anchor = self.fc_mm_1(anchor)
        anchor = self.fc_mm_2(anchor)
        anchor = self.fc_mm_3(anchor)
        anchor = self.fc_mm_4(anchor)
        anchor = self.fc_mm_5(anchor)

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

class BasicFC_GN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_GN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.gn = nn.GroupNorm(32, out_channels, eps=0.001)  # 32 is num groups

    def forward(self, x):
        x = self.fc(x)
        x = self.gn(x)
        return F.relu(x, inplace=True)

class BasicFC(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.fc(x)
        return F.relu(x, inplace=True)