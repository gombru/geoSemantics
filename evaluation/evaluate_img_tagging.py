# Image tagging evaluation
# Measures Accuracy at K, typically k=1,10
# Accuracy at k: Is one of the K top predicted hashtags in the vocab?

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
k = 10 # Compute accuracy at k (will also compute it at 1)
save_img = True # Save some random image tagging results


print("Reading tags embeddings ...")
tags_embeddings = aux.read_embeddings(tags_embeddings_path)
print("Reading imgs embeddings ...")
img_embeddings = aux.read_embeddings(img_embeddings_path)
print("Reading tags of testing images")
test_images_tags = aux.read_tags(test_split_path)



print("Starting per-image evaluation")

total_accuracy_at_1 = 0.0
total_accuracy_at_k = 0.0
for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):

    if i % 500 == 0: print(i)
    img_id = str(img_id)
    tags_distances = {}
    # Compute distance between the image and each tag
    for tag, tag_embedding in tags_embeddings.items():
        d = F.pairwise_distance(tag_embedding, img_embedding, p=2)
        tags_distances[tag] = d

    # Sort images by distance to tag
    tags_sorted_by_dist = sorted(tags_distances.values())
    tags_sorted_by_dist = tags_sorted_by_dist[0:k]

    # Save img
    if save_img and random.randint(0,100000) < 50:
        if not os.path.isdir(dataset + '/tagging_results/' + img_id + '/'):
            os.makedirs(dataset + '/tagging_results/' + img_id + '/')
        copyfile('../../../datasets/YFCC100M/test_img/' + img_id + '.jpg', dataset + '/retrieval_results/' + img_id + '/' + img_id + '.jpg')
        # Save txt file with gt and predicted tags
        with open(dataset + '/retrieval_results/' + img_id + '/tags.txt','w') as outfile:
            outfile.write('GT_tags\n')
            for tag in test_images_tags[img_id]:
                outfile.write(tag + ' ')
            outfile.write('\nPredicted_tags\n')
            for tag in tags_sorted_by_dist:
                outfile.write(tag[0] + ' ')

    # Compute Accuracy at 1
    if tags_sorted_by_dist[0][0] in test_images_tags[img_id]:
        total_accuracy_at_1 += 1
    # Compute Accuracy at k
    for tag in tags_sorted_by_dist:
        if tag[0] in test_images_tags[img_id]:
            total_accuracy_at_k += 1
            break

total_accuracy_at_1 /= len(img_embeddings)
total_accuracy_at_k /= len(img_embeddings)

print("Accuracy at 1:" + str(total_accuracy_at_1))
print("Accuracy at " + str(k) + " :" + str(total_accuracy_at_k))