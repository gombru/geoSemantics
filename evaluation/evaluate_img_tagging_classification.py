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
model_name = 'YFCC_MCLL_epoch_6_ValLoss_7.76'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
results_path = dataset + 'results/' + model_name + '/images_test.json'
accuracy_k = 10 # Compute accuracy at k (will also compute it at 1)
save_img = True # Save some random image tagging results


print("Reading results ...")
results = json.load(open(results_path))
print("Reading tags of testing images ... ")
test_images_tags = aux.read_tags(test_split_path)



print("Starting per-image evaluation")

total_accuracy_at_1 = 0.0
total_accuracy_at_k = 0.0
pdist = nn.PairwiseDistance(p=2)

for i, (img_id, img_result) in enumerate(results.items()):

    if i % 500 == 0: print(i)
    img_id = str(img_id)
    cur_img_tags = img_result['tags_indices']

    # Compute Accuracy at 1
    if cur_img_tags[0] in test_images_tags[int(img_id)]:
        total_accuracy_at_1 += 1
    # Compute Accuracy at k
    for cur_img_tag in cur_img_tags:
        if cur_img_tag in test_images_tags[int(img_id)]:
            total_accuracy_at_k += 1
            break

total_accuracy_at_1 /= len(results)
total_accuracy_at_k /= len(results)

print("Accuracy at 1:" + str(total_accuracy_at_1))
print("Accuracy at " + str(accuracy_k) + " :" + str(total_accuracy_at_k))