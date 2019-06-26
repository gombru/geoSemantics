from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import image_processing
from PIL import Image
import json
import random

class YFCC_Dataset(Dataset):

    def __init__(self, root_dir, split, random_crop, mirror):

        self.split = split
        self.random_crop = random_crop
        self.mirror = mirror
        self.tags = []

        if 'train' in self.split:
            self.root_dir = root_dir.replace('/hd/datasets/','/ssd2/') + 'train_img/'
        else:
            self.root_dir = root_dir.replace('/hd/datasets/', '/datasets/')  + 'val_img/'

        print("Loading tag list ...")
        tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
        for line in open(tags_file):
            self.tags.append(line.replace('\n',''))
        print("Vocabulary size: " + str(len(self.tags)))

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/splits/' + split))
        # self.num_elements = 100000
        print("Number of elements in " + split + ": " + str(self.num_elements))

        # Initialize containers
        self.img_ids = np.zeros(self.num_elements, dtype=np.uint64)
        self.images_tags = []

        # Read data
        print("Reading data ...")
        for i,line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
            if i % 1000000 == 0 and i != 0: print(i)
            # if i == 100000: break
            data = line.split(';')
            self.img_ids[i] = int(data[0])
            img_tags = data[1].split(',')
            img_tags_indices = []
            for img_tag in img_tags:
                img_tags_indices.append(self.tags.index(img_tag))
            self.images_tags.append(img_tags_indices)

        print("Data read. Set size: " + str(len(self.img_ids)))


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = '{}{}{}'.format(self.root_dir, self.img_ids[idx], '.jpg')

        # Load and transform image
        try:
            image = Image.open(img_name)
        except:
            new_img_name = '../../../ssd2/YFCC100M/train_img/6985418911.jpg'
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
            new_img_name = '../../../ssd2/YFCC100M/train_img/6985418911.jpg'
            print("Using hardcoded: " + new_img_name)
            image = Image.open(new_img_name)
            if self.random_crop != 0:
                image = image_processing.RandomCrop(image,self.random_crop)
            im_np = np.array(image, dtype=np.float32)
            im_np = image_processing.PreprocessImage(im_np)

        # Get target vector (multilabel classification)
        # target = np.zeros(100000, dtype=np.float32)
        # target[self.images_tags[idx]] = 1
        target_indices = np.ones(15, dtype=np.int)*100000
        target_indices[0:len(self.images_tags[idx])] = np.array(self.images_tags[idx]).astype(int)

        # Select a random image tag just to evaluate precision
        tag = random.choice(self.images_tags[idx])

        # Build tensors
        img_tensor = torch.from_numpy(np.copy(im_np))
        label = torch.from_numpy(np.array([tag]))
        label = label.type(torch.LongTensor)
        # target = torch.from_numpy(target)
        target_indices = torch.from_numpy(target_indices)


        return img_tensor, target_indices, label