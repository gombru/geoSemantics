import torch.nn as nn
import MyResNet
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=100000)
        ct = 0
        for child in self.cnn.children():
            ct += 1
            if ct < len(self.cnn.children()) - 1:
                print("Freezing: " + str(child))
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("Not Freezing: " + str(child))

    def forward(self, image):
        x = self.cnn(image)
        if not self.training:
        	x = F.sigmoid(x, dim=1)
        return x