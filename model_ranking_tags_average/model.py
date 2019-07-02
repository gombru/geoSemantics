import torch.nn as nn
import MyResNet
import torch
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, embedding_dims, margin, norm_degree):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=embedding_dims)
        self.c = {}
        self.c['margin'] = margin
        self.c['norm_degree'] = norm_degree

    def forward(self, image, tags_p, tags_n):
        x = self.cnn(image)

        # Check if triplet is already correct (not used for the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).cuda()
        d_i_tp = F.pairwise_distance(x, tags_p, p=self.c['norm_degree'])
        d_i_tn = F.pairwise_distance(x, tags_n, p=self.c['norm_degree'])

        for i in range(0,len(d_i_tp)):
            if (d_i_tn[i] - d_i_tp[i]) > self.c['margin']:
                correct[0] += 1

        return x, correct