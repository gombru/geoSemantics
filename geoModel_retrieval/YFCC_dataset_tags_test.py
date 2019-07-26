from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import numpy as np


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split

        # Load GenSim Word2Vec model
        print("Loading textual model ...")
        text_model_path = '../../../datasets/YFCC100M/' + '/vocab/vocab_100k.json'
        self.text_model = json.load(open(text_model_path))
        print("Vocabulary size: " + str(len(self.text_model)))

        # print("Normalizing vocab")
        # for k, v in self.text_model.items():
        #     v = np.asarray(v, dtype=np.float32)
        #     self.text_model[k] = v / np.linalg.norm(v, 2)

        self.tags_list = list(self.text_model.keys())

    def __len__(self):
        return len(self.tags_list)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding

    def __getitem__(self, idx):
        tag_str = str(self.tags_list[idx])
        tag = self.__getwordembedding__(tag_str)

        tag = torch.from_numpy(tag)
        lat = torch.from_numpy(np.array([0], dtype=np.float32))
        lon = torch.from_numpy(np.array([0], dtype=np.float32))

        return tag_str, tag, lat, lon