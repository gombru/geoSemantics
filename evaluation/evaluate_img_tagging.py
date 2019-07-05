# Image tagging evaluation
# Measures Accuracy at K, typically k=1,10
# Accuracy at k: Is one of the K top predicted hashtags in the vocab?

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
accuracy_k = 10 # Compute accuracy at k (will also compute it at 1)
save_img = False # Save some random image tagging results

measure = 'dotP' # 'distance'

distance_norm = 2 # 2
if measure == 'distance':
    print("Using pairwise distance with norm: " + str(distance_norm))

normalize = False # Normalize img embeddings and tag embeddings using L2 norm
print("Normalize tags and img embeddings: " + str(normalize))

print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))
print("Reading tags of testing images ... ")
test_images_tags = aux.read_tags(test_split_path)

if normalize:
    print("Using L2 normalization on img AND tag embeddings")

print("Puting tags embeddings in a tensor")
tags_embeddings_tensor = torch.zeros([len(tags_embeddings), embedding_dim], dtype=torch.float32).cuda()
tags = []
for i,(tag, tag_embedding) in enumerate(tags_embeddings.items()):
    tags.append(tag)
    tag_np_embedding = np.asarray(tag_embedding, dtype=np.float32)
    if normalize:
        tag_np_embedding /= np.linalg.norm(tag_np_embedding)
    tags_embeddings_tensor[i,:] = torch.from_numpy(tag_np_embedding)
del tags_embeddings

print("Starting per-image evaluation")

total_accuracy_at_1 = 0.0
total_accuracy_at_k = 0.0
dist = nn.PairwiseDistance(p=distance_norm)

for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):

    if i % 100 == 0 and i > 0:
        print(str(i) + ": Curr acc at " + str(accuracy_k) + " --> " + str(100*total_accuracy_at_k/i))
    img_id = str(img_id)
    img_np_embedding = np.asarray(img_embedding, dtype=np.float32)
    if normalize:
        img_np_embedding /= np.linalg.norm(img_np_embedding)
    img_embeddings_tensor = torch.from_numpy(img_np_embedding).cuda()

    if measure == 'distance':
        distances = dist(tags_embeddings_tensor, img_embeddings_tensor)
        # distances = np.array(distances.cpu())
        # indices_sorted = np.argsort(distances)[0:accuracy_k] # if distances
        indices_sorted = np.array(distances.sort(descending=False)[1][0:accuracy_k].cpu())

    elif measure == 'dotP':
        products = tags_embeddings_tensor.mm(img_embeddings_tensor.reshape(1,-1).t()).view(-1)
        products = F.softmax(products, dim=0)
        indices_sorted = np.array(products.sort(descending=True)[1][0:accuracy_k].cpu())

    else:
        print("Measure not found: " + str(measure))
        break

    # Compute Accuracy at 1
    correct = False
    correct_tag = ''
    if tags[indices_sorted[0]] in test_images_tags[int(img_id)]:
        total_accuracy_at_1 += 1
    # Compute Accuracy at k
    for idx in indices_sorted:
        if tags[idx] in test_images_tags[int(img_id)]:
            correct = True
            correct_tag = tags[idx]
            # print("Correct tag: " + tags[idx])
            total_accuracy_at_k += 1
            break

    # Save img
    if save_img and correct and random.randint(0,10) < 1:
        print("Saving. Correct tag: " + correct_tag)
        if not os.path.isdir(dataset + '/tagging_results/' + model_name + '/' + img_id + '/'):
            os.makedirs(dataset + '/tagging_results/' + model_name + '/' + img_id + '/')
        copyfile('../../../datasets/YFCC100M/test_img/' + img_id + '.jpg', dataset + '/tagging_results/' + model_name + '/' + img_id + '/' + img_id + '.jpg')
        # Save txt file with gt and predicted tags
        with open(dataset + '/tagging_results/' + model_name + '/' + img_id + '/tags.txt','w') as outfile:
            outfile.write('GT_tags\n')
            for tag in test_images_tags[int(img_id)]:
                outfile.write(tag + ' ')
            outfile.write('\nPredicted_tags\n')
            for idx in indices_sorted:
                outfile.write(tags[idx] + ' ')


total_accuracy_at_1 /= len(img_embeddings)
total_accuracy_at_k /= len(img_embeddings)

print("Accuracy at 1:" + str(total_accuracy_at_1*100))
print("Accuracy at " + str(accuracy_k) + " :" + str(total_accuracy_at_k*100))