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
        self.img_embeddings_dir = self.root_dir + 'img_embeddings/' + img_backbone_model + '/test/'
        self.split = split

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open(self.root_dir + 'splits/' + split))
        self.num_elements = 100
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)

        # Container for img embeddings
        self.img_embeddings = np.zeros((self.num_elements, 300), dtype=np.float32)

        # Read data
        print("Reading data ...")
        correct = 0
        errors = 0
        for i, line in enumerate(open(self.root_dir + 'splits/' + split)):
            if i % 100000 == 0 and i != 0: print(i)
            if i == 100: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            try:
                json_name = '{}{}{}'.format(self.img_embeddings_dir, self.img_ids[i], '.json')
                img_e = json.load(open(json_name))
                self.img_embeddings[i, :] = np.asarray(img_e[str(self.img_ids[i])], dtype=np.float32)
                correct += 1
            except:
                errors += 1
                self.img_embeddings[i, :] = np.zeros(300, dtype=np.float32)

        print("Data read. Set size: " + str(len(self.img_ids)))
        print("Correct: " + str(correct) + "; Errors: " + str(errors))



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img = self.img_embeddings[idx, :]

        # Build tensors
        img = torch.from_numpy(img)

        img_id = str(self.img_ids[idx])

        return img_id, img