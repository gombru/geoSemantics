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

        print("Loading tag|loc queries ...")
        queries_file = self.root_dir + 'geosensitive_queries/queries.txt'
        self.query_tags_names = []
        self.query_lats_str = []
        self.query_lons_str = []

        num_query_pairs = 500000

        self.query_tags_tensor = np.zeros([num_query_pairs, 300], dtype=np.float32)
        self.latitudes_tensor = np.zeros([num_query_pairs, 1], dtype=np.float32)
        self.longitudes_tensor = np.zeros([num_query_pairs, 1], dtype=np.float32)

        for i, line in enumerate(open(queries_file, 'r')):
            d = line.split(',')
            self.query_tags_names.append(d[0])
            lat = (float(d[1]) + 90) / 180
            lon = (float(d[2]) + 180) / 360
            self.query_lats_str.append(d[1])
            self.query_lons_str.append(d[2].replace('\n', ''))
            self.query_tags_tensor[i, :] = np.asarray(self.text_model[d[0]], dtype=np.float32)
            self.latitudes_tensor[i, :] = lat
            self.longitudes_tensor[i, :] = lon

    def __len__(self):
        return len(self.query_tags_names)

    def __getitem__(self, idx):

        tag = self.query_tags_tensor[idx, :]
        tag = torch.from_numpy(tag)
        tag_str = str(self.query_tags_names[idx])

        lat = self.latitudes_tensor[idx, :]
        lat = torch.from_numpy(lat)
        lat_str = str(self.query_lats_str[idx])

        lon = self.longitudes_tensor[idx, :]
        lon = torch.from_numpy(lon)
        lon_str = str(self.query_lons_str[idx])

        return tag, lat, lon, tag_str, lat_str, lon_str