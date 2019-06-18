# Image retrieval for every tag in vocabulary
# Only evaluate tag if at least K test images have it
# Measure Precision at K

import aux
import torch
import torch.nn.functional as F
import torch.nn as nn
import operator
import random
from shutil import copyfile
import os
import json
import numpy as np

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'YFCC_triplet_Img2Hash_e1024_m1_randomNeg_epoch_18_ValLoss_0.31'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
img_embeddings_path = dataset + 'results/' + model_name + '/images_test.json'
# tags_embeddings_path = dataset + 'results/' + model_name + '/tags.json'
# If using GloVe embeddings directly
print("Using GloVe embeddings")
tags_embeddings_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
embedding_dim = 300
precision_k = 10  # Compute precision at k
save_img = False  # Save some random image retrieval results

print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))
print("Reading tags of testing images ...")
test_images_tags = aux.read_tags(test_split_path)

print("Get tags with at least k appearances in test images")
tags_test_histogram = {}
for id, tags in test_images_tags.items():
    for tag in tags:
        if tag not in tags_test_histogram:
            tags_test_histogram[tag] = 1
        else:
            tags_test_histogram[tag] += 1

print("Total tags in test images: " + str(len(tags_test_histogram)))

print("Filtering vocab")
tags_test_histogram_filtered = {}
for k, v in tags_test_histogram.items():
    if v > precision_k:
        tags_test_histogram_filtered[k] = v

print("Total tags in test images with more than " + str(precision_k) + " appearances: " + str(
    len(tags_test_histogram_filtered)))

print("Puting image embeddings in a tensor")
# Put img embeddings in a tensor
img_embeddings_tensor = torch.zeros([len(img_embeddings), embedding_dim], dtype=torch.float32).cuda()
img_ids = []
for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):
    img_ids.append(img_id)
    img_embeddings_tensor[i, :] = torch.from_numpy(np.asarray(img_embedding, dtype=np.float32))
del img_embeddings

print("Starting per-tag evaluation")
pdist = nn.PairwiseDistance(p=2)
total_precision = 0.0
for i, (tag, test_appearances) in enumerate(tags_test_histogram_filtered.items()):
    if i % 500 == 0: print(str(i) + ': ' + tag)

    tag_embedding_tensor = torch.from_numpy(np.asarray(tags_embeddings[tag], dtype=np.float32)).cuda()
    distances = pdist(img_embeddings_tensor, tag_embedding_tensor)
    distances = np.array(distances.cpu())

    # Sort images by distance to tag
    indices_sorted = np.argsort(distances)[0:precision_k]

    # Save img
    if save_img and random.randint(0, len(tags_test_histogram_filtered)) < 2000:
        if not os.path.isdir(dataset + '/retrieval_results/model_name/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/model_name/' + tag + '/')

        for idx in indices_sorted:
            copyfile('../../../datasets/YFCC100M/test_img/' + img_ids[idx] + '.jpg',
                     dataset + '/retrieval_results/model_name/' + tag + '/' + img_ids[idx] + '.jpg')

    # Compute Precision at k
    precision_tag = 0.0
    for idx in indices_sorted:
        if tag in test_images_tags[int(img_ids[idx])]:
            precision_tag += 1

    precision_tag /= precision_k
    total_precision += precision_tag

total_precision /= len(tags_test_histogram_filtered)

print("Precision at " + str(precision_k*100) + ": " + str(total_precision*100))