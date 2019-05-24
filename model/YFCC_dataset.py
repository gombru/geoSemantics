from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import image_processing
from PIL import Image
import json
import random
# from gensim.models.keyedvectors import KeyedVectors

class YFCC_Dataset(Dataset):

    def __init__(self, root_dir, split, random_crop, mirror):

        self.split = split
        self.random_crop = random_crop
        self.mirror = mirror

        if 'train' in self.split:
            self.root_dir = root_dir.replace('/hd/datasets/','/ssd2/') + '/train_img/'
        else:
            self.root_dir = root_dir.replace('/hd/', '/datasets/')  + '/val_img/'

        # Load GenSim Word2Vec model
        print("Loading textual model ...")
        # text_model_path = '../../../datasets/YFCC100M/text_models/gensim_glove840B300d_vectors.txt'
        # self.text_model = KeyedVectors.load_word2vec_format(text_model_path, binary=False, unicode_errors='ignore')
        text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
        self.text_model = json.load(open(text_model_path))
        print("Vocabulary size: " + str(len(self.text_model)))

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/splits/' + split))
        # self.num_elements = 5000
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.tags = []
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.zeros(self.num_elements, dtype=np.float32)

        # Read data
        print("Reading data ...")
        for i,line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
            if i % 2000000 == 0 and i != 0: print(i)
            # if i == 5000: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            tags_array = data[1].split(',')
            self.tags.append(tags_array)
            self.latitudes[i] = float(data[4])
            self.longitudes[i] = float(data[5])

        print("Data read. Set size: " + str(len(self.tags)) )


    def __len__(self):
        return len(self.img_ids)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = np.asarray(self.text_model[tag], dtype=np.float32)
        return tag_embedding


    def __getitem__(self, idx):
        img_name = '{}/{}{}'.format(self.root_dir, self.img_ids[idx], '.jpg')

        # Load and transform image
        try:
            image = Image.open(img_name)
        except:
            new_img_name = '{}/{}{}'.format(self.root_dir, '6985418911', '.jpg')
            print("Img file " + img_name + " not found, using hardcoded " + new_img_name)
            image = Image.open(new_img_name)

        try:
            if self.random_crop != 0:
                image = image_processing.RandomCrop(image,self.random_crop)
            if self.mirror:
                image = image_processing.Mirror(image)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)

        except:
            print("Error in data aumentation with image " + img_name)
            new_img_name = '{}/{}{}'.format(self.root_dir, '6985418911', '.jpg')
            print("Using hardcoded: " + new_img_name)
            image = Image.open(new_img_name)
            if self.random_crop != 0:
                image = image_processing.RandomCrop(image,self.random_crop)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)

        # Select a random positive tag
        tag_pos = random.choice(self.tags[idx])
        tag_pos_embedding = self.__getwordembedding__(tag_pos)

        # Select a negative tag
        # --> Random negative: Random tag from random image
        while True:
            negative_img_idx = random.randint(0, self.num_elements - 1)
            tag_neg = random.choice(self.tags[negative_img_idx])
            if tag_neg not in self.tags[idx]:
                break
        tag_neg_embedding = self.__getwordembedding__(tag_neg)

        # Build tensors
        img_tensor = torch.from_numpy(np.copy(im_np))
        latitude = torch.from_numpy(np.array(self.latitudes[idx]))
        longitude = torch.from_numpy(np.array(self.longitudes[idx]))
        tag_pos_tensor = torch.from_numpy(tag_pos_embedding)
        tag_neg_tensor = torch.from_numpy(tag_neg_embedding)

        return img_tensor, tag_pos_tensor, tag_neg_tensor, latitude, longitude