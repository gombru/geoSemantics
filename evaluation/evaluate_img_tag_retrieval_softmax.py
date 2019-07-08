# Image retrieval for every tag in vocabulary
# Only evaluate tag if at least K test images have it
# Measure Precision at K


import aux
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from shutil import  copyfile
import os
import json
import numpy as np

dataset = '../../../datasets/YFCC100M/'
model_name = 'YFCC_MCLL_epoch_3_ValLoss_7.55'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
img_embeddings_path = dataset + 'results/' + model_name + '/images_embeddings_test.json'
tags_embeddings_path = dataset + 'results/' + model_name + '/tags_embeddings.json'
# If using GloVe embeddings directly
# print("Using GloVe embeddings")
# tags_embeddings_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
embedding_dim = 300
precision_k = 10  # Compute precision at k
save_img = False # Save some random image tagging results

normalize = False # Normalize img embeddings and tag embeddings using L2 norm
print("Normalize tags and img embeddings: " + str(normalize))

print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))
print("Reading tags of testing images ... ")
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
    if v >= precision_k:
        tags_test_histogram_filtered[k] = v

print("Total tags in test images with more than " + str(precision_k) + " appearances: " + str(
    len(tags_test_histogram_filtered)))

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
print("Shape tags: " + str(tags_embeddings_tensor.shape))

print("Putting image embeddings in a tensor")
# Put img embeddings in a tensor
img_embeddings_tensor = torch.zeros([len(img_embeddings), embedding_dim], dtype=torch.float32).cuda()
img_ids = []
for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):
    img_ids.append(img_id)
    img_np_embedding = np.asarray(img_embedding, dtype=np.float32)
    if normalize:
        img_np_embedding /= np.linalg.norm(img_np_embedding)
    img_embeddings_tensor[i, :] = torch.from_numpy(img_np_embedding)
del img_embeddings
print("Shape img em: " +str(img_embeddings_tensor.shape))

print("Computing images scores per tag and softmax")
products = img_embeddings_tensor.mm(tags_embeddings_tensor.t())
print("Shape products: " + str(products.shape))
products = F.softmax(products, dim=1)

top_imgs_tag = {}
print("Getting top img per tag")
for i in range(0,len(products)):
    indices_sorted = np.array(products[:,i].sort(descending=True)[1][0:precision_k].cpu()).tolist()
    top_imgs_tag[tags[i]] = indices_sorted


print("Starting per-tag evaluation")
total_precision = 0.0
for i, (tag, test_appearances) in enumerate(tags_test_histogram_filtered.items()):
    if i % 100 == 0 and i > 0:
        print(str(i) + ':  Cur P at ' + str(precision_k) + " --> " + str(100*total_precision/i))

    # Compute Precision at k
    correct = False
    precision_tag = 0.0
    for idx in top_imgs_tag[tag]:
        if tag in test_images_tags[int(img_ids[idx])]:
            correct = True
            precision_tag += 1

    precision_tag /= precision_k
    total_precision += precision_tag

    # Save img
    if save_img and correct and random.randint(0, 100) < 5:
        print("Saving results for: " + tag)
        if not os.path.isdir(dataset + '/retrieval_results/' + model_name + '/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/' + model_name + '/' + tag + '/')

        for idx in indices_sorted:
            copyfile('../../../datasets/YFCC100M/test_img/' + img_ids[idx] + '.jpg',
                     dataset + '/retrieval_results/' + model_name + '/' + tag + '/' + img_ids[idx] + '.jpg')

total_precision /= len(tags_test_histogram_filtered)

print("Precision at " + str(precision_k) + ": " + str(total_precision*100))