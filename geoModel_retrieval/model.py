import torch.nn as nn
import MyResNet
import torch
import torch.nn.functional as F
from collections import OrderedDict


class Model(nn.Module):

    def __init__(self, margin, norm_degree, CNN_checkpoint):
        super(Model, self).__init__()

        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=300)

        if CNN_checkpoint:
            print("Loading ResNet checkpoint: " + str(CNN_checkpoint))
            state_dict = torch.load(CNN_checkpoint, map_location={'cuda:0': 'cuda:2', 'cuda:1': 'cuda:2', 'cuda:3': 'cuda:2'})
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[11:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.cnn.load_state_dict(new_state_dict, strict=True)
        else:
            print("NOT loading CNN checkpoint")

        print("Freezing CNN")
        ct = 0
        for child in self.cnn.children():
            ct += 1
            if ct < len(list(self.cnn.children())):
                # print("Freezing: " + str(child))
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("Not Freezing: " + str(child))

        self.extra_net = MMNet()
        self.initialize_weights()
        self.c = {}
        self.c['margin'] = margin
        self.c['norm_degree'] = norm_degree

    def forward(self, img_p, img_n, tag, lat, lon):

        img_p = self.cnn(img_p)
        img_n = self.cnn(img_n)

        img_p, img_n, anchor = self.extra_net(img_p, img_n, tag, lat, lon)

        # Check if triplet is already correct (not used for the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).cuda()
        d_i_tp = F.pairwise_distance(anchor, img_p, p=self.c['norm_degree'])
        d_i_tn = F.pairwise_distance(anchor, img_n, p=self.c['norm_degree'])

        for i in range(0,len(d_i_tp)):
            if (d_i_tn[i] - d_i_tp[i]) > self.c['margin']:
                correct[0] += 1

        return img_p, img_n, anchor, correct


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

        # loc layers
        self.fc_loc = BasicFC_BN(2, 10)

        # mm tag|loc layers
        self.fc_mm_1 = BasicFC_BN(310, 1024)
        self.fc_mm_2 = BasicFC_BN(1024, 1024)
        self.fc_mm_3 = BasicFC_BN(1024, 1024)
        self.fc_mm_4 = nn.Linear(1024, 300)

    def forward(self, img_p, img_n, tag, lat, lon):
        # Images embeddings processing
        img_p = self.fc_i_1(img_p)
        img_p = self.fc_i_2(img_p)
        img_p = self.fc_i_3(img_p)

        img_n = self.fc_i_1(img_n)
        img_n = self.fc_i_2(img_n)
        img_n = self.fc_i_3(img_n)

        # Location
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # MM tag|loc
        anchor = torch.cat((loc, tag), dim=1)
        anchor = self.fc_mm_1(anchor)
        anchor = self.fc_mm_2(anchor)
        anchor = self.fc_mm_3(anchor)
        anchor = self.fc_mm_4(anchor)

        return img_p, img_n, anchor

class BasicFC_BN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001) # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)