import torch
import torch.nn as nn
import torch.nn.functional as F
import MyResNet
import math

class Model(nn.Module):

    def __init__(self, embedding_dims=1024, margin=1, norm_degree=2):
        super(Model, self).__init__()
        self.c={}
        self.c['embedding_dims'] = embedding_dims
        self.c['margin'] = margin
        self.c['norm_degree'] = norm_degree
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=self.c['embedding_dims'])
        self.extra_net = TextNet(self.c)
        self.initialize_weights()

    def forward(self, image, tag_pos, tag_neg, lat, lon):

        i_e = self.cnn(image)
        tp_e = self.extra_net(tag_pos)
        tn_e = self.extra_net(tag_neg)

        # Check if triplet is already correct (not used for the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).cuda()
        d_i_tp = F.pairwise_distance(i_e, tp_e, p=self.c['norm_degree'])
        d_i_tn = F.pairwise_distance(i_e, tn_e, p=self.c['norm_degree'])

        for i in range(0,len(d_i_tp)):
            if (d_i_tn[i] - d_i_tp[i]) > self.c['margin']:
                correct[0] += 1

        return i_e, tp_e, tn_e, correct

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

class TextNet(nn.Module):

    def __init__(self, c):
        super(TextNet, self).__init__()

        self.fc1 = BasicFC(300, 512)
        self.fc2 = BasicFC(512, c['embedding_dims'])


    def forward(self, tag):

        x = self.fc1(tag)
        x = self.fc2(x)
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