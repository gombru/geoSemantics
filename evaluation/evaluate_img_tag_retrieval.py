# Image retrieval for every tag in vocabulary
# Only evaluate tag if at least K test images have it
# Measure Precision at K

import aux
import torch.nn.functional as F
import operator
import random
from shutil import  copyfile
import os

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'YFCC_triplet_Img2Hash_e1024_m1_randomNeg'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
img_embeddings_path = dataset + model_name + '/results/' + 'images_test.txt'
tags_embeddings_path = dataset + model_name + '/results/' + 'tags.txt'
k = 10 # Compute precision at k
save_img = True # Save some random image retrieval results


print("Reading tags embeddings ...")
tags_embeddings = aux.read_embeddings(tags_embeddings_path)
print("Reading imgs embeddings ...")
img_embeddings = aux.read_embeddings(img_embeddings_path)
print("Reading tags of testing images")
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
for k,v in tags_test_histogram:
    if v > k: tags_test_histogram_filtered[k] = v

print("Total tags in test images with more than " + str(k) + " appearances: " + str(len(tags_test_histogram_filtered)))

print("Starting per-tag evaluation")

total_precision = 0.0
for i, (tag, test_appearances) in enumerate(tags_test_histogram_filtered.items()):

    if i % 500 == 0: print(i)

    tag_embedding = tags_embeddings[tag]
    image_distances = {}
    # Compute distance between the tag and each image
    for img_id, img_embedding in img_embeddings.items():
        d = F.pairwise_distance(tag_embedding, img_embedding, p=2)
        image_distances[img_id] = d

    # Sort images by distance to tag
    img_sorted_by_dist = sorted(image_distances.values())
    img_sorted_by_dist = img_sorted_by_dist[0:k]

    # Save img
    if save_img and random.randint(0,100000) < 50:
        if not os.path.isdir(dataset + '/retrieval_results/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/' + tag + '/')

        for img in img_sorted_by_dist:
            copyfile('../../../datasets/YFCC100M/test_img/' + img[0] + '.jpg', dataset + '/retrieval_results/' + tag + '/' + img[0] + '.jpg')

    # Compute Precision at k
    precision_tag = 0.0
    for img in img_sorted_by_dist:
        if tag in test_images_tags[img[0]]:
            precision_tag += 1

    precision_tag /= k
    total_precision += precision_tag

total_precision /= len(tags_test_histogram_filtered)

print("Precision at " + str(k) + " :" + str(total_precision))