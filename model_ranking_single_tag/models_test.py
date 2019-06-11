import torch
import torch.nn as nn
import torch.nn.functional as F
import MyResNet


class SuperTagsModel(nn.Module):

    def __init__(self, embedding_dims=1024):
        super(SuperTagsModel, self).__init__()
        self.extra_net = TagsModel(embedding_dims)

    def forward(self, tag):
        x = self.extra_net(tag)
        return x

class TagsModel(nn.Module):

    def __init__(self, embedding_dims):
        super(TagsModel, self).__init__()
        self.fc1 = BasicFC(300, 512)
        self.fc2 = BasicFC(512, embedding_dims)

    def forward(self, tag):
        x = self.fc1(tag)
        x = self.fc2(x)
        return x

class ImagesModel(nn.Module):

    def __init__(self, embedding_dims):
        super(ImagesModel, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=False, num_classes=embedding_dims)

    def forward(self, image):
        i_e = self.cnn(image)
        return i_e


class BasicFC(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)