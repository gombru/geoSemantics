from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import time


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir):

        self.root_dir = root_dir
        # Load GenSim Word2Vec model
        print("Loading textual model ...")
        text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
        self.text_model = json.load(open(text_model_path))
        print("Vocabulary size: " + str(len(self.text_model)))
        print("Normalizing vocab")
        for k, v in self.text_model.items():
            v = np.asarray(v, dtype=np.float32)
            self.text_model[k] = v / np.linalg.norm(v, 2)

        print("Loading tags list")
        self.tags_list = []
        tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
        for i, line in enumerate(open(tags_file)):
            tag = line.replace('\n', '').lower()
            self.tags_list.append(tag)

    def __len__(self):
        return len(self.tags_list)

    def __getitem__(self, idx):
        # print(str(idx) + ": " + str(self.tags_list[idx]))
        tag = self.text_model[self.tags_list[idx]]
        tag = torch.from_numpy(tag)
        return tag