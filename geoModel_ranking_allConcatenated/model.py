import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):

    def __init__(self, margin=1):
        super(Model, self).__init__()
        self.c={}
        self.c['margin'] = margin
        self.extra_net = MMNet_3()
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

class Model_Multiple_Negatives(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.extra_net = MMNet()
        self.initialize_weights()

    def forward(self, img, tag, lat, lon):
        score = self.extra_net(img, tag, lat, lon)
        return score

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



class Model_Test_Retrieval(nn.Module):

    def __init__(self):
        super(Model_Test_Retrieval, self).__init__()
        self.extra_net = MMNet_3()

    def forward(self, img, tag, lat, lon):
        # Here tag is [100kx300]
        # Others are [1xk], so I expand them
        img_batch = torch.zeros([len(tag), 300], dtype=torch.float32).cuda()
        lat_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda()
        lon_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda()
        img_batch[:,:] = img
        lat_batch[:,:] = lat_batch
        lon_batch[:,:] = lon_batch
        score = self.extra_net(img_batch, tag, lat_batch, lon_batch)
        return score


class Model_Test_Tagging(nn.Module):

    def __init__(self):
        super(Model_Test_Tagging, self).__init__()
        self.extra_net = MMNet_3()

    def forward(self, img, tag, lat, lon):
        # Here tag is [100kx300]
        # Others are [1xk], so I expand them
        img_batch = torch.zeros([len(tag), 300], dtype=torch.float32).cuda()
        lat_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda()
        lon_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda()
        img_batch[:,:] = img
        lat_batch[:,:] = lat_batch
        lon_batch[:,:] = lon_batch
        score = self.extra_net(img_batch, tag, lat_batch, lon_batch)
        return score


class MMNet(nn.Module):

    def __init__(self):
        super(MMNet, self).__init__()

        self.fc_i = BasicFC_BN(300,300)
        self.fc_t = BasicFC_BN(300,300)
        self.fc_loc = BasicFC_BN(2,10)

        self.fc1 = BasicFC(610, 512)
        self.fc2 = BasicFC(512, 512)
        self.fc3 = BasicFC(512, 512)
        self.fc4 = nn.Linear(512, 1)


    def forward(self, img, tag, lat, lon):

        # A layer with BN for each modality
        img = self.fc_i(img)
        tag = self.fc_t(tag)
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # Concatenate
        x = torch.cat((img, tag), dim=1)
        x = torch.cat((loc, x), dim=1)

        # MLP
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

class MMNet_2(nn.Module):

    def __init__(self):
        super(MMNet_2, self).__init__()

        self.fc_i = BasicFC_BN(300,300)
        self.fc_t = BasicFC_BN(300,300)
        self.fc_loc = BasicFC_BN(2,10)

        self.fc1 = BasicFC(610, 512)
        self.fc2 = BasicFC(512, 512)
        self.fc3 = BasicFC(512, 512)
        self.fc4 = BasicFC(512, 1024)
        self.fc5 = BasicFC(1024, 1024)
        self.fc6 = BasicFC(1024, 512)
        self.fc7 = nn.Linear(512, 1)


    def forward(self, img, tag, lat, lon):

        # A layer with BN for each modality
        img = self.fc_i(img)
        tag = self.fc_t(tag)
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # Concatenate
        x = torch.cat((img, tag), dim=1)
        x = torch.cat((loc, x), dim=1)

        # MLP
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        return x

class MMNet_3(nn.Module):

    def __init__(self):
        super(MMNet_3, self).__init__()

        self.fc_i = BasicFC_BN(300,300)
        self.fc_t = BasicFC_BN(300,300)
        self.fc_loc = BasicFC_BN(2,10)

        self.fc1 = BasicFC(610, 2048)
        self.fc2 = BasicFC(2048, 2048)
        self.fc3 = BasicFC(2048, 1024)
        self.fc4 = BasicFC(1024, 1024)
        self.fc5 = BasicFC(1024, 512)
        self.fc6 = BasicFC(512, 512)
        self.fc7 = nn.Linear(512, 1)


    def forward(self, img, tag, lat, lon):

        # A layer with BN for each modality
        img = self.fc_i(img)
        tag = self.fc_t(tag)
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # Concatenate
        x = torch.cat((img, tag), dim=1)
        x = torch.cat((loc, x), dim=1)

        # MLP
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        return x

class BasicFC(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        # self.bn = nn.BatchNorm1d(out_channels, eps=0.001) # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        # x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicFC_BN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001) # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)