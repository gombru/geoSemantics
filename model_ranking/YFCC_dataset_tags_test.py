from __future__ import print_function, division
import torch
import numpy as np
import json

class YFCC_Dataset_Tags_Test():

    def __init__(self):

        print("Loading textual model ...")
        text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
        self.text_model = json.load(open(text_model_path))
        self.num_elements = len(self.text_model)
        print("Vocabulary size: " + str(self.num_elements))

        print("Transforming text model data into arrays")
        self.tags = []
        self.tags_text_model = np.zeros((self.num_elements, 300), dtype=np.float32)
        for i, (k,v) in enumerate(self.text_model.items()):
            self.tags.append(k)
            self.tags_text_model[i,:] = np.asarray(v, dtype=np.float32)

        del self.text_model
        print("Dataset built")

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):

        tag = self.tags[idx]
        tag_tensor = torch.from_numpy(self.tags_text_model[idx,:])

        return tag, tag_tensor