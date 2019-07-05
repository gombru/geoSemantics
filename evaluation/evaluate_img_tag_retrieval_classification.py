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
model_name = 'YFCC_MCLL_epoch_3_ValLoss_7.55'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
top_img_per_tag_path = dataset + 'results/' + model_name + '/tags_top_img.json'

precision_k = 10  # Compute precision at k
save_img = True  # Save some random image retrieval results

normalize = True # Normalize img embeddings and tag embeddings using L2 norm

print("Loading tag list ...")
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
tags_list = []
for line in open(tags_file):
    tags_list.append(line.replace('\n', ''))
print("Vocabulary size: " + str(len(tags_list)))

print("Reading top img per class")
top_img_per_tag = json.load(open(top_img_per_tag_path))

print("Reading tags of testing images ...")
test_images_tags = aux.read_tags(test_split_path)
print("Num test images: " + str(len(test_images_tags)))

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
    if v >= precision_k:
        tags_test_histogram_filtered[k] = v

print("Total tags in test images with more than " + str(precision_k) + " appearances: " + str(
    len(tags_test_histogram_filtered)))


print("Starting per-tag evaluation")
total_precision = 0.0
for i, (tag, test_appearances) in enumerate(tags_test_histogram_filtered.items()):
    if i % 100 == 0 and i > 0:
        print(str(i) + ':  Cur P at ' + str(precision_k) + " --> " + str(100*total_precision/i))

    top_img_curTag = top_img_per_tag[str(tags_list.index(tag))]

    # Compute Precision at k
    correct = False
    precision_tag = 0.0
    for top_img_idx in top_img_curTag:
        if tag in test_images_tags[int(top_img_idx)]:
            correct = True
            precision_tag += 1

    precision_tag /= precision_k
    total_precision += precision_tag

    # Save img
    if save_img and correct and random.randint(0, 100) < 5:
        print("Saving results for: " + tag)
        if not os.path.isdir(dataset + '/retrieval_results/' + model_name + '/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/' + model_name + '/' + tag + '/')

        for top_img_idx in top_img_curTag:
            copyfile('../../../datasets/YFCC100M/test_img/' + str(top_img_idx) + '.jpg',
                     dataset + '/retrieval_results/' + model_name + '/' + tag + '/' + str(top_img_idx) + '.jpg')

total_precision /= len(tags_test_histogram_filtered)

print("Precision at " + str(precision_k) + ": " + str(total_precision*100))