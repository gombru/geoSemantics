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

        print("Loading tag|loc queries ...")
        queries_file = self.root_dir + 'geosensitive_queries/queries.txt'
        self.query_tags = []
        self.query_lats = []
        self.query_lons = []

        self.query_lats_str = []
        self.query_lons_str = []

        for line in open(queries_file, 'r'):
            d = line.split(',')
            self.query_tags.append(d[0])

            self.query_lats.append((float(d[1]) + 90) / 180)
            self.query_lons.append((float(d[2]) + 180) / 360)

            self.query_lats_str.append(d[1])
            self.query_lons_str.append(d[2].replace('\n',''))

        print("Number of queries: " + str(len(self.query_tags)))

    def __len__(self):
        return len(self.query_tags)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding

    def __getitem__(self, idx):

        tag_str = str(self.query_tags[idx])
        tag = self.__getwordembedding__(tag_str)

        tag = torch.from_numpy(tag)
        lat = torch.from_numpy(np.array([self.query_lats[idx]], dtype=np.float32))
        lon = torch.from_numpy(np.array([self.query_lons[idx]], dtype=np.float32))

        lat_str = self.query_lats_str[idx]
        lon_str = self.query_lons_str[idx]


        return tag_str, tag, lat, lon, lat_str, lon_str