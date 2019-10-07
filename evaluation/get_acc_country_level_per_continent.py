import aux
import random
from shutil import copyfile
import os
import json
import numpy as np
import geopy.distance


granularities = [750] # granularities in km country (750km)
continent_num_images = [9730619, 9627860, 2601501, 770204, 749160, 367584]
continent_labels = ['NA','EU','AS','OC','SA','AF']
continent_correct = np.zeros(6)
continent_total = np.zeros(6)

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

dataset = '../../../hd/datasets/YFCC100M/'
model_name = 'geoModel_ranking_allConcatenated_randomTriplets_MCLL_GN_TAGIMGL2_EML2_lr0_005_LocZeros_onlyImgNegTriplets_MCCTagEmbeddings_2ndTraining_epoch_15_ValLoss_0.02'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
top_img_per_tag_path = dataset + 'results/' + model_name + '/tagLoc_top_img.json'

precision_k = 10  # Compute precision at k
save_img = False  # Save some random image retrieval results


print("Loading tag list ...")
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
tags_list = []
for line in open(tags_file):
    tags_list.append(line.replace('\n', ''))
print("Vocabulary size: " + str(len(tags_list)))

print("Reading top img per class")
top_img_per_tag = json.load(open(top_img_per_tag_path))
print("Num query pairs: "+ str(len(top_img_per_tag.keys())))

print("Reading tags and locations of testing images ...")
test_images_tags, test_images_latitudes, test_images_longitudes = aux.read_tags_and_locations(test_split_path)
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
precisions = np.zeros(len(granularities), dtype=np.float32)
ignored = 0
used = 0
geolocator = Nominatim(user_agent="specify_your_app_name_here")
for i, (pair_info, cur_tag_top_img) in enumerate(top_img_per_tag.items()):

    d = pair_info.split(',')
    # print(d)
    cur_tag = d[0]
    cur_lat = float(d[1]) # * 180 - 90 # + 90) / 180
    cur_lon = float(d[2]) # * 360 -  180 #  180) / 360

    if cur_tag not in tags_test_histogram_filtered:
        ignored+=1
        continue

    used+=1

    # Get continent of current query
    location = geolocator.reverse(str(float(data[4])) + ", " + str(float(data[5])))
    country_code = location.raw['address']['country_code'].upper()
    try:
        continent_name = pc.country_alpha2_to_continent_code(country_code)
        continent_idx = continent_labels.index(continent_name)
        continent_total[continent_idx]+=1
    except:
        continent_idx = 10

    if i % 100 == 0 and used > 0:
        print(str(i) + ':  Cur P at ' + str(precision_k) + " --> " + str(100*precisions[0]/used))


    # Compute Precision at k
    correct = False
    precisions_tag = np.zeros(len(granularities), dtype=np.float32)
    for top_img_idx in cur_tag_top_img:
        if cur_tag in test_images_tags[int(top_img_idx)]:
            # The image has query tag. Now check its location!
            results = check_location(cur_lat, cur_lon, test_images_latitudes[int(top_img_idx)], test_images_longitudes[int(top_img_idx)])
            for r_i,r in enumerate(results):
                if r == 1:
                    precisions_tag[r_i] += 1

    precisions_tag /= precision_k

    if precisions_tag[3] > 0.2:
        correct = True

    if precisions_tag[0] > 0 and continent_idx < 10:
        continent_correct[continent_idx] += 1


    precisions += precisions_tag


precisions /= used
precisions *= 100

print("Used query pairs: " + str(used))
print("Ignored pairs: " + str(ignored))
print("Location Sensitive Precision at " + str(precision_k) + ":")

for i,p in enumerate(precisions):
    print(granularities_str[-i-1] + ': ' + str(p))

print(model_name)

print("Continent Labels")
print(continent_labels)
print("Continent Totals")
print(continent_total)
print("Continent Correct")
print(continent_correct)
print("Continent Precisions")
continent_precisions = continent_correct / continent_total
print(continent_precisions)
out_file_continents = '../../../datasets/YFCC100M/anns/continent_precisions.json'
out_data = {}
for i,l in enumerate(continent_labels):
    out_data[l] = continent_precisions[i]
json.dump(out_data, open(out_file_continents,'w'))