from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import time


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, img_backbone_model, split):

        self.root_dir = root_dir
        self.split = split
        self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + img_backbone_model + '/test.txt'

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open(self.root_dir + 'splits/' + split))
        # self.num_elements = 100
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.img_embeddings = np.zeros((self.num_elements, 300), dtype=np.float32)

        # Read data
        print("Reading split data ...")
        for i, line in enumerate(open(self.root_dir + 'splits/' + split)):
            if i % 2000000 == 0 and i != 0: print(i)
            if i == 100: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            self.latitudes[i] = float(data[4])
            self.longitudes[i] = float(data[5])
            # Coordinates normalization
            self.latitudes[i] = (self.latitudes[i] + 90) / 180
            self.longitudes[i] = (self.longitudes[i] + 180) / 360

        print("Data read. Set size: " + str(len(self.img_ids)))
        print("Latitudes min and max: " + str(min(self.latitudes)) + ' ; ' + str(max(self.latitudes)))

        print("Reading image embeddings")
        img_em_c = 0
        for i, line in enumerate(open(self.img_embeddings_path)):
            if i % 2000000 == 0 and i != 0: print(i)
            # if i == 10000: break
            img_em_c+=1
            d = line.split(',')
            img_id = int(d[0])
            img_idx, = np.where(self.img_ids == img_id)
            self.img_embeddings[img_idx, :] = np.asarray(d[1:], dtype=np.float32)

        print("Img embeddings loaded: " + str(img_em_c))


    def __len__(self):
        return len(self.img_ids)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding


    def __getitem__(self, idx):

        img = self.img_embeddings[idx, :]
        lat = self.latitudes[idx]
        lon = self.longitudes[idx]

        # Build tensors
        img = torch.from_numpy(img)
        lat = torch.from_numpy(np.array([lat]))
        lon = torch.from_numpy(np.array([lon]))

        img_id = str(self.img_ids[idx])

        return img_id, img, lat, lon