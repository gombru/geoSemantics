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
import geopy.distance

def check_location(lat1,lon1,lat2,lon2):
    coords_1 = (lat1,lon1)
    coords_2 = (lat2, lon2)

    try:
        distance_km = geopy.distance.vincenty(coords_1, coords_2).km
    except:
        print("Error computing distance with gropy. Values:")
        print(coords_1)
        print(coords_2)
        distance_km = 100000

    results = np.zeros(len(granularities))
    for i,gr in enumerate(granularities):
        if distance_km <= gr:
            results[i] = 1
        else:
            break
    return results

granularities = [99999999,2500,750,200,25,1] # granularities in km
granularities_str = ['street level (1km)', 'city (25km)', 'region (200km)', 'country (750km)', 'continent (2500km)', 'not sensitive (inf)']


dataset = '../../../hd/datasets/YFCC100M/'
queries_file = dataset + 'geosensitive_queries/queries.txt'
model_name = 'YFCC_NCSL_3rdtraining_epoch_3_ValLoss_0.37'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
img_embeddings_path = dataset + 'results/' + model_name + '/images_embeddings_test.json'
# tags_embeddings_path = dataset + 'results/' + model_name + '/tags_embeddings.json'
# If using GloVe embeddings directly 370111
print("Using GloVe embeddings")
tags_embeddings_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
embedding_dim = 300
precision_k = 10  # Compute precision at k
save_img = False  # Save some random image retrieval results

measure = 'distance' # 'distance', 'cosineSim', 'dotP'

distance_norm = 1 # 2
if measure == 'distance':
    print("Using pairwise distance with norm: " + str(distance_norm))

normalize = True # Normalize img embeddings and tag embeddings using L2 norm
print("Normalize tags and img embeddings: " + str(normalize))

print("Reading tags embeddings ...")
tags_embeddings = json.load(open(tags_embeddings_path))
print("Reading imgs embeddings ...")
img_embeddings = json.load(open(img_embeddings_path))
print("Reading tags of testing images ...")
test_images_tags, test_images_latitudes, test_images_longitudes = aux.read_tags_and_locations(test_split_path)
print("Num test images read: " + str(len(test_images_tags)))

if normalize:
    print("Using L2 normalization on img AND tag embeddings")

print("Reading queries")
query_tags = []
query_lats = []
query_lons = []
for line in open(queries_file,'r'):
    d = line.split(',')
    query_tags.append(d[0])
    query_lats.append(float(d[1]))
    query_lons.append(float(d[2]))
print("Number of queries: " + str(len(query_tags)))

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

print("Starting per-tag evaluation")
dist = nn.PairwiseDistance(p=distance_norm)
cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)

precisions = np.zeros(len(granularities), dtype=np.float32)
ignored = 0
used = 0
for i, cur_tag in enumerate(query_tags):

    if cur_tag not in tags_test_histogram_filtered:
        ignored+=1
        continue

    used+=1

    if i % 500 == 0 and i > 0:
        print(str(i) + ':  Cur P at ' + str(precision_k) + " --> " + str(100*precisions[0]/i))
        print(precisions)

    tag_np_embedding = np.asarray(tags_embeddings[cur_tag], dtype=np.float32)
    if normalize:
        tag_np_embedding /= np.linalg.norm(tag_np_embedding)

    tag_embedding_tensor = torch.from_numpy(tag_np_embedding).cuda()

    if measure == 'distance':
        # print("Shapes")
        # print(img_embeddings_tensor.shape)
        # print(tag_embedding_tensor.shape)
        distances = dist(img_embeddings_tensor, tag_embedding_tensor)
        indices_sorted = np.array(distances.sort(descending=False)[1][0:precision_k].cpu())

    elif measure == 'cosineSim':
        similarities = cosSim(img_embeddings_tensor, tag_embedding_tensor)
        indices_sorted = np.array(distances.sort(descending=True)[1][0:precision_k].cpu())


    # Need to apply softmax though images scores for each tag!
    # elif measure == 'dotP':
    #     products = img_embeddings_tensor.mm(tag_embedding_tensor.reshape(1,-1).t()).view(-1)
    #     indices_sorted = np.array(products.sort(descending=True)[1][0:precision_k].cpu())

    else:
        print("Measure not found: " + str(measure))
        break


    # Compute Precision at k
    correct = False
    precisions_tag = np.zeros(len(granularities), dtype=np.float32)
    for top_img_idx in indices_sorted:
        if cur_tag in test_images_tags[int(img_ids[top_img_idx])]:
            # The image has query tag. Now check its location!
            results = check_location(query_lats[i], query_lons[i], test_images_latitudes[int(img_ids[top_img_idx])], test_images_longitudes[int(img_ids[top_img_idx])])
            for r_i,r in enumerate(results):
                if r == 1:
                    precisions_tag[r_i] += 1

    precisions_tag /= precision_k

    if precisions_tag[3] > 0.2:
        correct = True

    precisions += precisions_tag

    # Save img
    if save_img and correct and random.randint(0, 100) < 5:
        print("Saving results for: " + tag)
        if not os.path.isdir(dataset + '/retrieval_results/' + model_name + '/' + tag + '/'):
            os.makedirs(dataset + '/retrieval_results/' + model_name + '/' + tag + '/')

        for idx in indices_sorted:
            copyfile('../../../datasets/YFCC100M/test_img/' + img_ids[idx] + '.jpg',
                     dataset + '/retrieval_results/' + model_name + '/' + tag + '/' + img_ids[idx] + '.jpg')


precisions /= used
precisions *= 100

print("Used query pairs: " + str(used))
print("Ignored pairs: " + str(ignored))
print("Location Sensitive Precision at " + str(precision_k) + ":")

for i,p in enumerate(precisions):
    print(granularities_str[-i-1] + ': ' + str(p))