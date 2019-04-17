from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import image_processing
from PIL import Image
import random
from gensim.models import Word2Vec

class YFCC_Dataset(Dataset):

    def __init__(self, root_dir, split, random_crop, mirror):

        self.root_dir = root_dir
        self.split = split
        self.random_crop = random_crop
        self.mirror = mirror

        # Load GenSim Word2Vec model
        text_model_path = root_dir + 'GoogleNews-vectors-negative300.bin'
        self.text_model = Word2Vec.load(text_model_path, binary=True)

        # Count number of elements
        self.num_elements = sum(1 for line in open(root_dir + 'splits/' + split))
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.tags = []
        self.latitudes = np.zeros(self.num_elements, dtype=np.float32)
        self.longitudes = np.empty(self.num_elements, dtype=np.float32)

        # Read data
        print("Reasing data ...")
        for i,line in enumerate(open(root_dir + 'splits/' + split)):
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            tags_array = data[1].split(',')
            self.tags[i].append(tags_array)
            self.latitudes[i] = float(data[4])
            self.longitudes[i] = float(data[5])

        print("Data read.")


    def __len__(self):
        return len(self.tweet_ids)

    def __getwordembedding__(self, tag):
        tag = tag.lower()
        tag_embedding = self.text_model[tag]
        return tag_embedding


    def __getitem__(self, idx):
        img_name = '{}{}/{}{}'.format(self.root_dir, 'img', self.img_ids[idx], '.jpg')

        # Load and transform image
        try:
            image = Image.open(img_name)
        except:
            new_img_name = '{}{}/{}{}'.format(self.root_dir, 'img', '000', '.jpg')
            print("Img file " + img_name + " not found, using hardcoded " + new_img_name)
            image = Image.open(new_img_name)

        try:
            if self.random_crop != 0:
                image = image_processing.RandomCrop(image,self.RandomCrop)
            if self.mirror:
                image = image_processing.Mirror(image)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)

        except:
            print("Error in data aumentation with image " + img_name)
            new_img_name = '{}{}/{}{}'.format(self.root_dir, 'img', '000', '.jpg')
            print("Using hardcoded: " + new_img_name)
            image = Image.open(new_img_name)
            if self.random_crop != 0:
                image = image_processing.RandomCrop(image,self.RandomCrop)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)

        # Select a random positive tag
        tag_pos = random.choice(self.tags[idx])
        tag_pos_embedding = self.__getwordembedding__(tag_pos)

        # Select a negative tag
        # --> Random negative: Random tag from random image
        negative_img = random.randint(0, self.num_elements)
        tag_neg_embedding = self.__getwordembedding__(tag_neg)




        # Build tensors
        img_tensor = torch.from_numpy(np.copy(im_np))
        latitude = torch.from_numpy(np.array(self.latitudes[idx]))
        longitude = torch.from_numpy(np.array(self.longitudes[idx]))
        tag_pos_tensor = torch.from_numpy(tag_pos_embedding)
        tag_neg_tensor = torch.from_numpy(tag_neg_embedding)

        return img_tensor, tag_pos_tensor, tag_neg_tensor, latitude, longitude