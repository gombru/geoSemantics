# Image tagging evaluation
# Measures Accuracy at K, typically k=1,10
# Accuracy at k: Is one of the K top predicted hashtags in the vocab?

import aux
import torch
import torch.nn as nn
import random
from shutil import  copyfile
import os
import json
import numpy as np

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'YFCC_NCSL_epoch_15_ValLoss_0.42.pth'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
img_embeddings_path = dataset + 'results/' + model_name + '/images_test.json'
# tags_embeddings_path = dataset + 'results/' + model_name + '/tags.json'
# If using GloVe embeddings directly
print("Using GloVe embeddings")
tags_embeddings_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
embedding_dim = 300
accuracy_k = 10 # Compute accuracy at k (will also compute it at 1)
save_img = True # Save some random image tagging results


print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))
print("Reading tags of testing images ... ")
test_images_tags = aux.read_tags(test_split_path)


print("Puting tags embeddings in a tensor")
# Put img embeddings in a tensor
tags_embeddings_tensor = torch.zeros([len(tags_embeddings), embedding_dim], dtype=torch.float32).cuda()
tags = []
for i,(tag, tag_embedding) in enumerate(tags_embeddings.items()):
    tags.append(tag)
    tags_embeddings_tensor[i,:] = torch.from_numpy(np.asarray(tag_embedding, dtype=np.float32))
del tags_embeddings

print("Starting per-image evaluation")

total_accuracy_at_1 = 0.0
total_accuracy_at_k = 0.0
pdist = nn.PairwiseDistance(p=2)

for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):

    if i % 500 == 0: print(i)
    img_id = str(img_id)

    img_embeddings_tensor = torch.from_numpy(np.asarray(img_embedding, dtype=np.float32)).cuda()
    distances = pdist(tags_embeddings_tensor, img_embeddings_tensor)
    distances = np.array(distances.cpu())

    # Sort tags by distance to image
    indices_sorted = np.argsort(distances)[0:accuracy_k]

    # Save img
    if save_img and random.randint(0,100000) < 1000:
        if not os.path.isdir(dataset + '/tagging_results/' + img_id + '/'):
            os.makedirs(dataset + '/tagging_results/' + img_id + '/')
        copyfile('../../../datasets/YFCC100M/test_img/' + img_id + '.jpg', dataset + '/tagging_results/' + img_id + '/' + img_id + '.jpg')
        # Save txt file with gt and predicted tags
        with open(dataset + '/tagging_results/' + img_id + '/tags.txt','w') as outfile:
            outfile.write('GT_tags\n')
            for tag in test_images_tags[int(img_id)]:
                outfile.write(tag + ' ')
            outfile.write('\nPredicted_tags\n')
            for idx in indices_sorted:
                outfile.write(tags[idx] + ' ')

    # Compute Accuracy at 1
    if tags[indices_sorted[0]] in test_images_tags[int(img_id)]:
        total_accuracy_at_1 += 1
    # Compute Accuracy at k
    for idx in indices_sorted:
        if tags[idx] in test_images_tags[int(img_id)]:
            total_accuracy_at_k += 1
            break

total_accuracy_at_1 /= len(img_embeddings)
total_accuracy_at_k /= len(img_embeddings)

print("Accuracy at 1:" + str(total_accuracy_at_1*100))
print("Accuracy at " + str(accuracy_k) + " :" + str(total_accuracy_at_k*100))