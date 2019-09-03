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
        queries_file = dataset_folder + 'geosensitive_queries/queries.txt'
        self.query_tags_names = []
        self.query_lats_str = []
        self.query_lons_str = []

        query_tags_tensor = np.zeros([num_query_pairs, 300], dtype=np.float32)
        latitudes_tensor = np.zeros([num_query_pairs, 1], dtype=np.float32)
        longitudes_tensor = np.zeros([num_query_pairs, 1], dtype=np.float32)

        for i, line in enumerate(open(queries_file, 'r')):
            d = line.split(',')
            query_tags_names.append(d[0])
            lat = (float(d[1]) + 90) / 180
            lon = (float(d[2]) + 180) / 360
            query_lats_str.append(d[1])
            query_lons_str.append(d[2].replace('\n', ''))
            query_tags_tensor[i, :] = np.asarray(text_model[d[0]], dtype=np.float32)
            latitudes_tensor[i, :] = lat
            longitudes_tensor[i, :] = lon

    def __len__(self):
        return len(self.query_tags_names)

    def __getitem__(self, idx):

        tag = query_tags_tensor[idx, :]
        tag = torch.from_numpy(tag)
        tag_str = query_tags_names[idx]

        lat = latitudes_tensor[idx, :]
        lat = torch.from_numpy(lat)
        lat_str = query_lats_str[idx]

        lon = longitudes_tensor[idx, :]
        lon = torch.from_numpy(lon)
        lon_str = query_lons_str[idx]

        return tag, lat, lon, tag_str, lat_str, lon_str