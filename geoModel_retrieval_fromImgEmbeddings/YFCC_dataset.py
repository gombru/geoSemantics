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
            # self.num_elements = 102400
        elif 'val' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/val.txt'
            # self.num_elements = 2048 * 2
        else:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/test.txt'

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

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open(self.root_dir + '/splits/' + split))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.tags = []
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.img_embeddings = {}

        # Read data
        print("Reading split data ...")
        for i, line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
            if i % 2000000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            tags_array = data[1].split(',')
            self.tags.append(tags_array)

            # self.latitudes[i] = float(data[4])
            # self.longitudes[i] = float(data[5])
            # # Coordinates normalization
            # self.latitudes[i] = (self.latitudes[i] + 90) / 180
            # self.longitudes[i] = (self.longitudes[i] + 180) / 360

        print("Data read. Set size: " + str(len(self.tags)))

        print("Latitudes min and max: " + str(min(self.latitudes)) + ' ; ' + str(max(self.latitudes)))
        print("Longitudes min and max: " + str(min(self.longitudes)) + ' ; ' + str(max(self.longitudes)))

        print("Reading image embeddings")
        img_em_c = 0
        for i, line in enumerate(open(self.img_embeddings_path)):
            if i % 100000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            img_em_c += 1
            d = line.split(',')
            img_id = int(d[0])
            img_em = np.asarray(d[1:], dtype=np.float32)
            # img_em = img_em / np.linalg.norm(img_em, 2)
            self.img_embeddings[img_id] = img_em
        print("Img embeddings loaded: " + str(img_em_c))

    def __len__(self):
        return len(self.img_ids)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding

    def __getitem__(self, idx):

        try:
            img_p = self.img_embeddings[self.img_ids[idx]]
        except:
            print("Couldn't find img embedding for image: " + str(self.img_ids[idx]) + ". Using 0s. " + str(idx))
            img_p = np.zeros(300, dtype=np.float32)

        # Select a random positive tag
        tag_str = random.choice(self.tags[idx])
        tag = self.__getwordembedding__(tag_str)
        lat = self.latitudes[idx]
        lon = self.longitudes[idx]

        #### Negatives selection
        ### Random negative imagen (not sharing the selected tag)
        while True:
            img_n_index = random.randint(0, self.num_elements - 1)
            if img_n_index != idx and tag_str not in self.tags[img_n_index]:
                break

        ### Random negative image. USE THIS when using location!!
        # while True:
        #     img_n_index = random.randint(0, self.num_elements - 1)
        #     if img_n_index != idx:
        #         break


        try:
            img_n = self.img_embeddings[self.img_ids[img_n_index]]
        except:
            print("Couldn't find img embedding for image: " + str(self.img_ids[idx]) + ". Using 0s. " + str(idx))
            img_p = np.zeros(300, dtype=np.float32)

        # Build tensors
        img_p = torch.from_numpy(np.copy(img_p))
        img_n = torch.from_numpy(np.copy(img_n))
        tag = torch.from_numpy(tag)
        lat = torch.from_numpy(np.array([lat], dtype=np.float32))
        lon = torch.from_numpy(np.array([lon], dtype=np.float32))

        return img_p, img_n, tag, lat, lon