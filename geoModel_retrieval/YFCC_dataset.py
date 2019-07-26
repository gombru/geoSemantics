from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model
import numpy as np
from PIL import Image
import image_processing
import geopy.distance



class YFCC_Dataset(Dataset):
    def __init__(self, root_dir, split, random_crop, mirror):

        self.root_dir = root_dir
        self.split = split
        self.random_crop = random_crop
        self.mirror = mirror
        self.distance_thresholds = [2500, 750, 200, 25, 1]
        self.current_threshold = 0

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

        if 'train' in self.split:
            self.root_dir = root_dir.replace('/hd/datasets/','/ssd2/') + 'train_img/'
            # self.num_elements = 650*500
        else:
            self.root_dir = root_dir.replace('/hd/datasets/', '/datasets/')  + 'val_img/'
            # self.num_elements = 650*10

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/splits/' + split))

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

            self.latitudes[i] = float(data[4])
            self.longitudes[i] = float(data[5])
            # Coordinates normalization
            self.latitudes[i] = (self.latitudes[i] + 90) / 180
            self.longitudes[i] = (self.longitudes[i] + 180) / 360

        print("Data read. Set size: " + str(len(self.tags)))

        print("Latitudes min and max: " + str(min(self.latitudes)) + ' ; ' + str(max(self.latitudes)))
        print("Longitudes min and max: " + str(min(self.longitudes)) + ' ; ' + str(max(self.longitudes)))


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
            distance_km = 100000

        return distance_km


    def __getitem__(self, idx):

        img_name = '{}{}{}'.format(self.root_dir, self.img_ids[idx], '.jpg')
        try:
            img_p = Image.open(img_name)
        except:
            new_img_name = '../../../ssd2/YFCC100M/train_img/6985418911.jpg'
            print("Img file " + img_name + " not found, using hardcoded " + new_img_name)
            img_p = Image.open(new_img_name)

        try:
            if self.random_crop != 0:
                img_p = image_processing.RandomCrop(img_p,self.random_crop)
            if self.mirror:
                img_p = image_processing.Mirror(img_p)
            img_p = np.array(img_p, dtype=np.float32)
            img_p = image_processing.PreprocessImage(img_p)

        except:
            print("Error in data aumentation with image " + img_name)
            new_img_name = '../../../ssd2/YFCC100M/train_img/6985418911.jpg'
            print("Using hardcoded: " + new_img_name)
            img_p = Image.open(new_img_name)
            if self.random_crop != 0:
                img_p = image_processing.RandomCrop(img_p,self.random_crop)
            img_p = np.array(img_p, dtype=np.float32)
            img_p = image_processing.PreprocessImage(img_p)

        # Select a random positive tag
        tag_str = random.choice(self.tags[idx])
        tag = self.__getwordembedding__(tag_str)
        lat = self.latitudes[idx]
        lon = self.longitudes[idx]


        #### Negatives selection

        ### Random negative imagen (not sharing the selected tag)
        # while True:
        #     img_n_index = random.randint(0, self.num_elements - 1)
        #     if img_n_index != idx and tag_str not in self.tags[img_n_index]:
        #         break

        ### Random negative image. USE THIS when using location!!
        negative_type = random.randint(0,1)

        if negative_type == 0: # Select a random negative

            while True:
                img_n_index = random.randint(0, self.num_elements - 1)
                if img_n_index != idx and tag_str not in self.tags[img_n_index]:
                    break

        else: # Select an image with the same tag but another location (more distant than a threshold)

            while True:
                img_n_index = random.randint(0, self.num_elements - 1)
                # Check that image shares the tag and it's not the same
                if tag_str in self.tags[img_n_index] and img_n_index != idx:
                    # Check that the distance is above distance threshold
                    locations_distance = self.__getdistance__(self.latitudes[idx], self.longitudes[idx], self.latitudes[img_n_index], self.longitudes[img_n_index])
                    if locations_distance > self.distance_thresholds[self.current_threshold]:
                        break


        img_name = '{}{}{}'.format(self.root_dir, self.img_ids[img_n_index], '.jpg')
        try:
            img_n = Image.open(img_name)
        except:
            new_img_name = '../../../ssd2/YFCC100M/train_img/6985418911.jpg'
            print("Img file " + img_name + " not found, using hardcoded " + new_img_name)
            img_n = Image.open(new_img_name)
        try:
            if self.random_crop != 0:
                img_n = image_processing.RandomCrop(img_n,self.random_crop)
            if self.mirror:
                img_n = image_processing.Mirror(img_n)
            img_n = np.array(img_n, dtype=np.float32)
            img_n = image_processing.PreprocessImage(img_n)

        except:
            print("Error in data aumentation with image " + img_name)
            new_img_name = '../../../ssd2/YFCC100M/train_img/6985418911.jpg'
            print("Using hardcoded: " + new_img_name)
            img_n = Image.open(new_img_name)
            if self.random_crop != 0:
                img_n = image_processing.RandomCrop(img_n,self.random_crop)
            img_n = np.array(img_n, dtype=np.float32)
            img_n = image_processing.PreprocessImage(img_n)

        # Build tensors
        img_p = torch.from_numpy(np.copy(img_p))
        img_n = torch.from_numpy(np.copy(img_n))
        tag = torch.from_numpy(tag)
        lat = torch.from_numpy(np.array([lat], dtype=np.float32))
        lon = torch.from_numpy(np.array([lon], dtype=np.float32))


        return img_p, img_n, tag, lat, lon