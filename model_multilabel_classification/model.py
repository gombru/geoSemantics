import torch.nn as nn
import MyResNet
import torch

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=100000)
        # ct = 0
        # for child in self.cnn.children():
        #     ct += 1
        #     if ct < len(list(self.cnn.children())):
        #         print("Freezing: " + str(child))
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         print("Not Freezing: " + str(child))

    def forward(self, image):
        x = self.cnn(image)
        # if not self.training:
        #   x = torch.sigmoid(x)
        return x


class Model_Test(nn.Module):

    def __init__(self):
        super(Model_Test, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=100000)

    def forward(self, image):
        x = self.cnn(image)
        x = torch.sigmoid(x)
        norm = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(norm)
        return x