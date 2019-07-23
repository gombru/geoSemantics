from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import numpy as np

class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, split, img_backbone_model):

        self.root_dir = root_dir
        self.split = split
        self.img_backbone_model = img_backbone_model

        if 'train' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/train_filtered.txt'
            self.num_elements = 102400
        elif 'val' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/val.txt'
            self.num_elements = 2048 * 2
        else:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/test.txt'



        self.img_embeddings = {}

        print("Reading image embeddings")
        img_em_c = 0
        for i, line in enumerate(open(self.img_embeddings_path)):
            if i % 100000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            img_em_c += 1
            d = line.split(',')
            img_id = int(d[0])
            img_em = np.asarray(d[1:], dtype=np.float32)
            img_em = img_em / np.linalg.norm(img_em, 2)
            self.img_embeddings[img_id] = img_em
        print("Img embeddings loaded: " + str(img_em_c))


    def __len__(self):
        return len(self.img_embeddings)


    def __getitem__(self, idx):

        try:
            img = self.img_embeddings[self.img_ids[idx]]
        except:
            print("Couldn't find img embedding for image: " + str(self.img_ids[idx]) + ". Using 0s. " + str(idx))
            img = np.zeros(300, dtype=np.float32)

        img_id = str(self.img_ids[idx])
        img = torch.from_numpy(np.copy(img))

        return img_id, img