import aux
import torch
import torch.nn as nn
import random
from shutil import  copyfile
import os
import json
import numpy as np


out_file = open('results/tagging_results.txt','w')

img_embeddings_path = 'results/images_embeddings.json'
# tags_embeddings_path = dataset + 'results/' + model_name + '/tags.json'
# If using GloVe embeddings directly
print("Using GloVe embeddings")
tags_embeddings_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
embedding_dim = 300
num_tags = 1000

normalize = True # Normalize img embeddings and tag embeddings using L2 norm

print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))

if normalize:
    print("Using L2 normalization on img AND tag embeddings")

print("Puting tags embeddings in a tensor")
# Put img embeddings in a tensor
tags_embeddings_tensor = torch.zeros([len(tags_embeddings), embedding_dim], dtype=torch.float32).cuda()
tags = []
for i,(tag, tag_embedding) in enumerate(tags_embeddings.items()):
    tags.append(tag)
    tag_np_embedding = np.asarray(tag_embedding, dtype=np.float32)
    if normalize:
        tag_np_embedding /= np.linalg.norm(tag_np_embedding)
    tags_embeddings_tensor[i,:] = torch.from_numpy(tag_np_embedding)
del tags_embeddings

print("Starting per-image tagging")

dist = nn.PairwiseDistance(p=2)

for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):

    img_id = str(img_id)
    img_np_embedding = np.asarray(img_embedding, dtype=np.float32)
    if normalize:
        img_np_embedding /= np.linalg.norm(img_np_embedding)
    img_embeddings_tensor = torch.from_numpy(img_np_embedding).cuda()
    distances = dist(tags_embeddings_tensor, img_embeddings_tensor)
    distances = np.array(distances.cpu())

    # Sort tags by distance to image
    indices_sorted = np.argsort(distances)[0:num_tags] # if distances

    out_file.write(img_id)
    for idx in indices_sorted:
        out_file.write(' ,' + tags[idx])
    out_file.write('\n')

print("DONE")