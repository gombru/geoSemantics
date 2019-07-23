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
model_name = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
top_img_per_tag_path = dataset + 'results/' + model_name + '/tags_top_img.json'
precision_k = 10  # Compute precision at k
save_img = False  # Save some random image retrieval results


print("Reading tags of testing images ...")
test_images_tags, test_images_latitudes, test_images_longitudes = aux.read_tags_and_locations(test_split_path)
print("Num test images read: " + str(len(test_images_tags)))

print("Reading top img per class")
top_img_per_tag = json.load(open(top_img_per_tag_path))

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

print("Loading tag list ...")
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
tags_list = []
for line in open(tags_file):
    tags_list.append(line.replace('\n', ''))
print("Vocabulary size: " + str(len(tags_list)))

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

    top_img_curTag = top_img_per_tag[str(tags_list.index(tag))]

    # Compute Precision at k
    correct = False
    precisions_tag = np.zeros(len(granularities), dtype=np.float32)
    for top_img_idx in top_img_curTag:
        if cur_tag in test_images_tags[int(top_img_idx)]:
            # The image has query tag. Now check its location!
            results = check_location(query_lats[i], query_lons[i], test_images_latitudes[int(top_img_idx)], test_images_longitudes[int(top_img_idx)])
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

        for idx in top_img_curTag:
            copyfile('../../../datasets/YFCC100M/test_img/' + str(idx) + '.jpg',
                     dataset + '/retrieval_results/' + model_name + '/' + tag + '/' + str(idx) + '.jpg')


precisions /= used
precisions *= 100

print("Used query pairs: " + str(used))
print("Ignored pairs: " + str(ignored))
print("Location Sensitive Precision at " + str(precision_k) + ":")

for i,p in enumerate(precisions):
    print(granularities_str[-i-1] + ': ' + str(p))