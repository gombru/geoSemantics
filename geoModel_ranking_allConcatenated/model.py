import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):

    def __init__(self, margin=1):
        super(Model, self).__init__()
        self.c={}
        self.c['margin'] = margin
        self.extra_net = MMNet(self.c)
        self.initialize_weights()

    def forward(self, img_p, tag_p, lat_p, lon_p, img_n, tag_n, lat_n, lon_n):

        score_p = self.extra_net(img_p, tag_p, lat_p, lon_p)
        score_n = self.extra_net(img_n, tag_n, lat_n, lon_n)


        # Check if pair is already correct (not used for the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).cuda()
        for i in range(0,len(score_p)):
            if (score_p[i] - score_n[i]) > self.c['margin']:
                correct[0] += 1

        return score_p, score_n, correct

    def initialize_weights(self):
        for l in self.extra_net.modules(): # Initialize only extra_net weights
            if isinstance(l, nn.Conv2d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                l.weight.data.normal_(0, math.sqrt(2. / n))
                if l.bias is not None:
                    l.bias.data.zero_()
            elif isinstance(l, nn.BatchNorm2d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                l.weight.data.normal_(0, 0.01)
                l.bias.data.zero_()

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