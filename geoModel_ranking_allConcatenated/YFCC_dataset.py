from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
import model

class YFCC_Dataset(Dataset):

    def __init__(self, root_dir, split, img_backbone_model):

        self.root_dir = root_dir
        self.split = split
        self.img_backbone_model = img_backbone_model
        self.gpu = 0
        self.margin = 1
        self.num_triplets_evaluated = 64  # Number of triplets to evaluate to then pick the hardest negative
        self.checkpoint = False

        # Load model to generate negative pairs
        print("Loading model to generate negative pairs")
        self.model = model.Model(margin=self.margin).cuda(self.gpu)
        checkpoint = torch.load(self.checkpoint, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})
        model.load_state_dict(checkpoint, strict=False)

        if 'train' in self.split:
            self.root_dir += 'image_embeddings/' + self.img_backbone_model + '/train_img/'
        else:
            self.root_dir += 'image_embeddings/' + self.img_backbone_model + '/val_img/'

        # Load GenSim Word2Vec model
        print("Loading textual model ...")
        text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
        self.text_model = json.load(open(text_model_path))
        print("Vocabulary size: " + str(len(self.text_model)))

        # Count number of elements
        print("Opening dataset ...")
        self.num_elements = sum(1 for line in open('../../../datasets/YFCC100M/splits/' + split))
        # self.num_elements = 100
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
            # if i == 100: break
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

    def __get_random_negative_triplet__(self, idx, img_n, tag_n, lat_n, lon_n):
        # Select randomly the element to change
        element_picker = random.randint(0, 2)

        if element_picker == 0:  # Change image
            negative_img_idx = random.randint(0, self.num_elements - 1)
            img_name = '{}{}{}'.format(self.root_dir, self.img_ids[negative_img_idx], '.json')
            img_n = json.load(open(img_name))
            img_n = np.asarray(img_n[self.img_ids[negative_img_idx]], dtype=np.float32)

        elif element_picker == 1:  # Change tag
            while True:  # Check that image does not have the randlomly selected tag
                negative_img_idx = random.randint(0, self.num_elements - 1)
                cur_tag_neg = random.choice(self.tags[negative_img_idx])
                if cur_tag_neg not in self.tags[idx]:
                    break
            tag_n = self.__getwordembedding__(cur_tag_neg)

        else:  # Change location
            negative_location_idx = random.randint(0, self.num_elements - 1)
            lat_n = self.latitudes[negative_location_idx]
            lon_n = self.longitudes[negative_location_idx]

        return img_n, tag_n, lat_n, lon_n


def __getitem__(self, idx):
        img_name = '{}{}{}'.format(self.root_dir, self.img_ids[idx], '.json')

        img_p= json.load(open(img_name))
        img_p = np.asarray(img_p[self.img_ids[idx]], dtype=np.float32)
        # Select a random positive tag
        tag_p = random.choice(self.tags[idx])
        tag_p = self.__getwordembedding__(tag_p)
        lat_p = self.latitudes[idx]
        lon_p = self.longitudes[idx]

        # Preset all elements of the negative triplet as the ones in the positive one
        img_n = img_p
        tag_n = tag_p
        lat_n = lat_p
        lon_n = lon_p

        #### Negatives selection
        ### Random negative selection
        if random.randint(0, 1) == 0:
            img_n, tag_n, lat_n, lon_n = self.get_random_negative_triplet(idx, img_n, tag_n, lat_n, lon_n)

        #### Hard negative selection
        else:
            # Initiliaze batch containers
            img_batch = np.zeros((self.num_triplets_evaluated, 300), dtype=np.float32)
            tag_batch = np.zeros((self.num_triplets_evaluated, 300), dtype=np.float32)
            lat_batch = np.zeros((self.num_triplets_evaluated, 1), dtype=np.float32)
            lon_batch = np.zeros((self.num_triplets_evaluated, 1), dtype=np.float32)

            # Load self.num_triplets_evaluated random triplets
            for x in range(self.num_triplets_evaluated):
                cur_img_n, cur_tag_n, cur_lat_n, cur_lon_n = self.get_random_negative_triplet(idx, img_n, tag_n, lat_n, lon_n)
                img_batch[0,:] = cur_img_n
                tag_batch[0,:] = cur_tag_n
                lat_batch[0,:] = cur_lat_n
                lon_batch[0,:] = cur_lon_n

            # Convert np arrays to tensors and varis
            img_var = torch.autograd.Variable(torch.from_numpy(img_batch))
            tag_var = torch.autograd.Variable(torch.from_numpy(tag_batch))
            lat_var = torch.autograd.Variable(torch.from_numpy(lat_batch))
            lon_var = torch.autograd.Variable(torch.from_numpy(lon_batch))

            # Compute output
            with torch.no_grad(): # don't compute gradient
                model.eval() # don't use dropout or batchnorm
                s_p, s_n, correct = model(img_var, tag_var, lat_var, lon_var, img_var, tag_var, lat_var, lon_var)

            # Get the triplet with maximum score
            top_scored_triplet_batch_idx = np.argsort(s_p.cpu())[-1]
            img_n = img_batch[top_scored_triplet_batch_idx,:]
            tag_n = tag_batch[top_scored_triplet_batch_idx,:]
            lat_n = lat_batch[top_scored_triplet_batch_idx,:]
            lon_n = lon_batch[top_scored_triplet_batch_idx,:]

        # Build tensors
        img_p = torch.from_numpy(img_p)
        tag_p = torch.from_numpy(tag_p)
        lat_p = torch.from_numpy(np.array(lat_p))
        lon_p = torch.from_numpy(np.array(lon_p))

        img_n = torch.from_numpy(img_n)
        tag_n = torch.from_numpy(tag_n)
        lat_n = torch.from_numpy(np.array(lat_n))
        lon_n = torch.from_numpy(np.array(lon_n))



        return img_p, tag_p, lat_p, lon_p, img_n, tag_n, lat_n, lon_n