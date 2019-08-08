from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import geopy.distance
import numpy as np
import math


class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, split, img_backbone_model):

        self.root_dir = root_dir
        self.split = split
        self.img_backbone_model = img_backbone_model
        self.distance_thresholds = [2500, 750, 200, 25, 1]
        self.current_threshold = 1

        if 'train' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/train_filtered.txt'
            images_per_tag_file = '../../../datasets/YFCC100M/' + 'splits/images_per_tag_train_filtered.json'
            self.num_elements = 1024 * 1000
        elif 'val' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/val.txt'
            images_per_tag_file = '../../../datasets/YFCC100M/' + 'splits/images_per_tag_val.json'
            self.num_elements = 1024 * 100
        else:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/test.txt'

        # Load img ids per tag
        print("Loading img ids per tag ...")
        self.images_per_tag = json.load(open(images_per_tag_file))

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
        # self.num_elements = sum(1 for line in open(self.root_dir + '/splits/' + split))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.tags = []
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.latitudes_or = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes_or = np.zeros(self.num_elements, dtype=np.float32)
        self.img_embeddings = {}
        self.img_ids2idx_map = {}

        # Read data
        print("Reading split data ...")
        for i, line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
            if i % 2000000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            tags_array = data[1].split(',')
            self.tags.append(tags_array)

            self.latitudes_or[i] = float(data[4])
            self.longitudes_or[i] = float(data[5])
            # Coordinates normalization
            self.latitudes[i] = (self.latitudes_or[i] + 90) / 180
            self.longitudes[i] = (self.longitudes_or[i] + 180) / 360

            self.img_ids2idx_map[int(data[0])] = i


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

    def __getdistance__(self, lat1, lon1, lat2, lon2):
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        try:
            distance_km = geopy.distance.vincenty(coords_1, coords_2).km
        except:
            print("Error computing distance with gropy. Values:")
            print(coords_1)
            print(coords_2)
            distance_km = 0
        return distance_km

    def __getdistanceFast__(self, lat1, lon1, lat2, lon2):
        deglen = 110.25
        x = lat1 - lat2
        y = (lon1 - lon2)*math.cos(lat2)
        return deglen*math.sqrt(x*x + y*y)

    def __getItemNotSharingTag__(self, idx, tag_str):
        while True:
            img_n_index = random.randint(0, self.num_elements - 1)
            if img_n_index != idx and tag_str not in self.tags[img_n_index]:
                break
        return img_n_index

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

        negative_type = random.randint(0, 1)
        negative_type = 0

        if negative_type == 0:  # Select a random negative
            img_n_index = self.__getItemNotSharingTag__(idx, tag_str)

        else:  # Select an image with the same tag but another location (more distant than a threshold)
            img_with_cur_tag = self.images_per_tag[tag_str]
            if isinstance(img_with_cur_tag, list):
                num_img_with_cur_tag = len(img_with_cur_tag)
            else:
                num_img_with_cur_tag = 0

            dist_checked = 0

            while True:
                if num_img_with_cur_tag < 2 or dist_checked > num_img_with_cur_tag or dist_checked == 100:
                    img_n_index = self.__getItemNotSharingTag__(idx, tag_str)
                    break
                try:
                    img_n_index = self.img_ids2idx_map[random.choice(img_with_cur_tag)]
                except:
                    img_n_index = self.__getItemNotSharingTag__(idx, tag_str)
                    break

                # Check that image it's not the same
                if img_n_index != idx:
                    # Check that the distance is above distance threshold
                    dist_checked += 1
                    locations_distance = self.__getdistanceFast__(self.latitudes_or[idx], self.longitudes_or[idx],
                                                                  self.latitudes_or[img_n_index],
                                                                  self.longitudes_or[img_n_index])
                    if locations_distance > self.distance_thresholds[self.current_threshold]:
                        break

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