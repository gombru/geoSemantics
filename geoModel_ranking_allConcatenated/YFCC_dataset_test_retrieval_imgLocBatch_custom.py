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

        print("Loading tag queries ... ")
        self.query_tags_names = []
        queries_file = root_dir + 'custom_tag_queries.txt'
        for line in open(queries_file, 'r'):
            self.query_tags_names.append(line.replace('\n',''))


        print("Loading loc queries ... ")
        self.tagLoc_queries = {}
        locs_file = root_dir + 'custom_loc_queries.txt'
        query_idx = 0
        for line in open(locs_file, 'r'):
            d = line.split(',')
            loc_str = d[0]
            lat = (float(d[1]) + 90) / 180
            lon = (float(d[2]) + 180) / 360
            for tag in self.query_tags_names:
                self.tagLoc_queries[query_idx] = {}
                self.tagLoc_queries[query_idx]['loc_str'] = loc_str
                self.tagLoc_queries[query_idx]['lat'] = lat
                self.tagLoc_queries[query_idx]['lon'] = lon
                self.tagLoc_queries[query_idx]['tag_str'] = tag
                query_idx += 1


    def __len__(self):
        return len(self.tagLoc_queries)

    def __getitem__(self, idx):

        query_idx = idx

        tag_str = self.tagLoc_queries[query_idx]['tag_str']
        tag = torch.from_numpy(self.text_model[tag_str])

        lat = self.tagLoc_queries[query_idx]['lat']
        lat = torch.from_numpy(lat)

        self.tagLoc_queries[query_idx]['lon']
        lon = torch.from_numpy(lon)

        loc_str = self.tagLoc_queries[query_idx]['loc_str']

        return tag, lat, lon, tag_str, loc_str
