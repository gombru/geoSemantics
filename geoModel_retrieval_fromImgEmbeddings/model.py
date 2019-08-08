import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, margin, norm_degree):
        super(Model, self).__init__()
        self.mm_net = NormLocEmGN()
        self.img_net = ImgNet()
        self.initialize_weights()
        self.c = {}
        self.c['margin'] = margin
        self.c['norm_degree'] = norm_degree

    def forward(self, img_p, img_n, tag, lat, lon):

        anchor = self.mm_net(tag, lat, lon)
        img_p = self.img_net(img_p)
        img_n = self.img_net(img_n)

        # Check if triplet is already correct (not used fors the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).cuda()
        d_i_tp = F.pairwise_distance(anchor, img_p, p=self.c['norm_degree'])
        d_i_tn = F.pairwise_distance(anchor, img_n, p=self.c['norm_degree'])

        for i in range(0, len(d_i_tp)):
            if (d_i_tn[i] - d_i_tp[i]) > self.c['margin']:
                correct[0] += 1

        return img_p, img_n, anchor, correct

    def initialize_weights(self):
        for l in self.mm_net.modules():  # Initialize only extra_net weights
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

        for l in self.img_net.modules():  # Initialize only extra_net weights
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



class NormLocEmGN(nn.Module):
    def __init__(self):
        super(NormLocEmGN, self).__init__()

        # loc layers
        self.fc_loc_1 = BasicFC(2, 10)
        self.fc_loc_2 = BasicFC(10, 10)
        self.fc_loc_3 = BasicFC(10, 10)

        # mm tag|loc layers
        self.fc_mm_1 = BasicFC_GN(310, 1024)
        self.fc_mm_2 = BasicFC_GN(1024, 2048)
        self.fc_mm_3 = BasicFC_GN(2048, 2048)
        self.fc_mm_4 = BasicFC_GN(2048, 1024)
        self.fc_mm_5 = nn.Linear(1024, 300)

    def forward(self, tag, lat, lon):

        # Location
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc_1(loc)
        loc = self.fc_loc_2(loc)
        loc = self.fc_loc_3(loc)
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

class NoNormLocEmGN(nn.Module):
    def __init__(self):
        super(NormLocEmGN, self).__init__()

        # loc layers
        self.fc_loc_1 = BasicFC(2, 10)
        self.fc_loc_2 = BasicFC(10, 10)
        self.fc_loc_3 = BasicFC(10, 10)

        # mm tag|loc layers
        self.fc_mm_1 = BasicFC_GN(310, 1024)
        self.fc_mm_2 = BasicFC_GN(1024, 2048)
        self.fc_mm_3 = BasicFC_GN(2048, 2048)
        self.fc_mm_4 = BasicFC_GN(2048, 1024)
        self.fc_mm_5 = nn.Linear(1024, 300)

    def forward(self, tag, lat, lon):

        # Location
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc_1(loc)
        loc = self.fc_loc_2(loc)
        loc = self.fc_loc_3(loc)

        # MM tag|loc
        anchor = torch.cat((loc, tag), dim=1)
        anchor = self.fc_mm_1(anchor)
        anchor = self.fc_mm_2(anchor)
        anchor = self.fc_mm_3(anchor)
        anchor = self.fc_mm_4(anchor)
        anchor = self.fc_mm_5(anchor)

        return anchor

class ExplicitLoc(nn.Module):
    def __init__(self):
        super(ExplicitLoc, self).__init__()

        # loc layers
        self.fc_loc_1 = BasicFC(2, 10)
        self.fc_loc_2 = BasicFC(10, 10)
        self.fc_loc_3 = BasicFC(10, 2)

        # mm tag|loc layers
        self.fc_tag_1 = BasicFC_GN(300, 1024)
        self.fc_tag_2 = BasicFC_GN(1024, 2048)
        self.fc_tag_3 = BasicFC_GN(2048, 1024)
        self.fc_tag_4 = nn.Linear(1024, 298)

    def forward(self, tag, lat, lon):

        # Location
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc_1(loc)
        loc = self.fc_loc_2(loc)
        loc = self.fc_loc_3(loc)

        # Tag
        tag = self.fc_tag_1(tag)
        tag = self.fc_tag_2(tag)
        tag = self.fc_tag_3(tag)
        tag = self.fc_tag_4(tag)

        # tag|loc
        anchor = torch.cat((loc, tag), dim=1)

        return anchor


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