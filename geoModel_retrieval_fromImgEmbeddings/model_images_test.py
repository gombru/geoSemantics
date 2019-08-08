import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.img_net = ImgNet()

    def forward(self, img):

        img = self.img_net(img)
        return img

class ImgNet(nn.Module):
    def __init__(self):
        super(ImgNet, self).__init__()
        self.fc_i_1 = BasicFC_GN(300, 1024)
        self.fc_i_2 = BasicFC_GN(1024, 2048)
        self.fc_i_3 = BasicFC_GN(2048, 1024)
        self.fc_i_4 = nn.Linear(1024, 300)

    def forward(self, img):
        img = self.fc_i_1(img)
        img = self.fc_i_2(img)
        img = self.fc_i_3(img)
        img = self.fc_i_4(img)
        return img



class BasicFC_BN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001) # momentum = 0.0001

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