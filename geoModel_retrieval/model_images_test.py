import torch.nn as nn
import MyResNet
import torch
import torch.nn.functional as F
from collections import OrderedDict


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=False, num_classes=300)
        self.extra_net = MMNet()


    def forward(self, img):

        img = self.cnn(img)
        img = self.extra_net(img)
        return img


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
    def __init__(self):
        super(MMNet, self).__init__()

        # img layers
        self.fc_i_1 = BasicFC_BN(300, 1024)
        self.fc_i_2 = nn.Linear(1024, 1024)
        self.fc_i_3 = nn.Linear(1024, 300)


    def forward(self, img_p):
        # Images embeddings processing
        img_p = self.fc_i_1(img_p)
        img_p = self.fc_i_2(img_p)
        img_p = self.fc_i_3(img_p)
        return img_p

class BasicFC_BN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001) # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)