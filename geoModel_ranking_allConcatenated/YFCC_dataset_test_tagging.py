from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import time


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, split, central_crop):

        self.root_dir = root_dir
        self.split = split

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open(self.root_dir + 'splits/' + split))
        self.num_elements = 10000
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.zeros(self.num_elements, dtype=np.float32)

        # Container for img embeddings
        self.img_embeddings = np.zeros((self.num_elements, 300), dtype=np.float32)

        # Read data
        print("Reading data ...")
        correct = 0
        errors = 0
        for i, line in enumerate(open(self.root_dir + 'splits/' + split)):
            if i % 100000 == 0 and i != 0: print(i)
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            self.latitudes[i] = float(data[4])
            self.longitudes[i] = float(data[5])
            # Coordinates normalization
            self.latitudes[i] = (self.latitudes[i] + 90) / 180
            self.longitudes[i] = (self.longitudes[i] + 180) / 360
            try:
                json_name = '{}{}{}'.format(self.root_dir, self.img_ids[i], '.json')
                img_e = json.load(open(json_name))
                self.img_embeddings[i, :] = np.asarray(img_e[str(self.img_ids[i])], dtype=np.float32)
                correct += 1
            except:
                errors += 1
                self.img_embeddings[i, :] = np.zeros(300, dtype=np.float32)
                self.latitudes[i] = 0.0
                self.longitudes[i] = 0.0

        print("Data read. Set size: " + str(len(self.tags)))
        print("Correct: " + str(correct) + "; Errors: " + str(errors))

        print("Latitudes min and max: " + str(min(self.latitudes)) + ' ; ' + str(max(self.latitudes)))
        print("Longitudes min and max: " + str(min(self.longitudes)) + ' ; ' + str(max(self.longitudes)))

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