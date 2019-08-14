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
        self.num_negatives = 6  # Number of negatives per anchor

        if 'train' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/train_filtered.txt'
            # self.num_elements = 1024 * 500
        elif 'val' in self.split:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/val.txt'
            # self.num_elements = 1024 * 200
        else:
            self.img_embeddings_path = self.root_dir + 'img_embeddings_single/' + self.img_backbone_model + '/test.txt'

        # Load GenSim Word2Vec model
        print("Loading textual model ...")
        text_model_path = '../../../datasets/YFCC100M/' + '/vocab/vocab_100k.json'
        self.text_model = json.load(open(text_model_path))
        print("Vocabulary size: " + str(len(self.text_model)))
        print("Normalizing vocab")
        for k, v in self.text_model.items():
            v = np.asarray(v, dtype=np.float32)
            self.text_model[k] = v / np.linalg.norm(v, 2)
        self.tags_list = list(self.text_model.keys())

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/' + '/splits/' + split))
        print("Number of elements in " + split + ": " + str(self.num_elements))

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

        print("Reading image embeddings")
        img_em_c = 0
        for i, line in enumerate(open(self.img_embeddings_path)):
            if i % 100000 == 0 and i != 0: print(i)
            if i == self.num_elements: break
            img_em_c += 1
            d = line.split(',')
            img_id = int(d[0])
            img_em = np.asarray(d[1:], dtype=np.float32)
            img_em = img_em / np.linalg.norm(img_em, 2)
            img_em[img_em != img_em] = 0
            self.img_embeddings[img_id] = img_em
        print("Img embeddings loaded: " + str(img_em_c))

    def __len__(self):
        return len(self.img_ids)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding

    def __get_random_negative_triplet__(self, idx, img_n, tag_n, lat_n, lon_n, tag_str):

        # Select randomly the element to change

        # element_picker = random.randint(0, 2)
        element_picker = random.randint(0, 1)

        if element_picker == 0:  # Change image
            while True:
                negative_img_idx = random.randint(0, self.num_elements - 1)
                if negative_img_idx != idx and tag_str not in self.tags[negative_img_idx]:
                    break
                negative_img_idx = random.randint(0, self.num_elements - 1)
            try:
                img_n = self.img_embeddings[self.img_ids[negative_img_idx]]
            except:
                print("Couldn't find img embedding for negative image: " + str(
                    self.img_ids[negative_img_idx]) + ". Using 0s." + str())
                img_n = np.zeros(300, dtype=np.float32)

        elif element_picker == 1:  # Change tag
            while True:  # Check that image does not have the randlomly selected tag
                cur_tag_neg = random.choice(self.tags_list)
                if cur_tag_neg not in self.tags[idx]:
                    break
            tag_n = self.__getwordembedding__(cur_tag_neg)

        # TODO: Here I should check that the selected location is farer than a TH
        else:  # Change location
            negative_location_idx = random.randint(0, self.num_elements - 1)
            lat_n = self.latitudes[negative_location_idx]
            lon_n = self.longitudes[negative_location_idx]

        # TODO: I should also create hard triplets with a negative image sharing a tag but with a different location.

        return img_n, tag_n, lat_n, lon_n

    def __getitem__(self, idx):

        # Initialize tensors container. In position 0 I put the anchor element. Others are negatives
        # But we initialize all with anchor info
        images = np.zeros((self.num_negatives + 1, 300), dtype=np.float32)
        tags = np.zeros((self.num_negatives + 1, 300), dtype=np.float32)
        latitudes = np.zeros((self.num_negatives + 1, 1), dtype=np.float32)
        longitudes = np.zeros((self.num_negatives + 1, 1), dtype=np.float32)

        try:
            images[:,:] = self.img_embeddings[self.img_ids[idx]]
        except:
            print("Couldn't find img embedding for image: " + str(self.img_ids[idx]) + ". Using 0s. " + str(idx))
            images[:, :] = np.zeros(300, dtype=np.float32)

        # Select a random positive tag
        tag_p = random.choice(self.tags[idx])
        tags[:] = self.__getwordembedding__(tag_p)

        latitudes[:] = self.latitudes[idx]
        longitudes[:] = self.longitudes[idx]


        #### Negatives selection
        ### Multiple Random negative selection
        for n_i in range(1,self.num_negatives+1):
            images[n_i,:], tags[n_i,:], latitudes[n_i], longitudes[n_i] = self.__get_random_negative_triplet__(idx, images[0,:], tags[0,:], latitudes[0], longitudes[0], tag_p)

        # Build tensors
        images = torch.from_numpy(images)
        tags = torch.from_numpy(tags)
        latitudes = torch.from_numpy(latitudes)
        longitudes = torch.from_numpy(longitudes)

        return images, tags, latitudes, longitudes