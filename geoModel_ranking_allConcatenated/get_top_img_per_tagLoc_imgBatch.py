# Get top images for tag-location pairs
# I will generate the query tag-location pairs before hand
# I will create a pair per test image (total 500k), chossing a random image tag and image location

import os
import torch.utils.data
import model
import json
import numpy as np
import YFCC_dataset_test_retrieval_imgLocBatch
import random
import time

random.seed(0)

dataset_folder = '../../../hd/datasets/YFCC100M/'
split = 'test.txt'
img_backbone_model = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55'

batch_size = 1
workers = 0

num_test_img = 200000
num_query_pairs = 1000
print("Using num query paris: " + str(num_query_pairs))

model_name = 'geoModel_ranking_allConcatenated_randomTriplets6Neg_MCLL_GN_TAGIMGL2_EML2_lr0_005_withLoc_2ndTraining_epoch_228_ValLoss_0.015.pth'
model_name = model_name.replace('.pth','')

gpus = [2]
gpu = 2

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/tagLoc_top_img_imgLocBatch.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0':'cuda:2', 'cuda:1':'cuda:2', 'cuda:3':'cuda:2'})


model_test = model.Model_Test_Retrieval_ImgBatch()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test_retrieval_imgLocBatch.YFCC_Dataset(dataset_folder)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)


imgs_tensor = np.zeros([num_test_img, 300], dtype=np.float32)
img_ids = []
print("Reading image embeddings")
for i, line in enumerate(open(dataset_folder + 'img_embeddings_single/' + img_backbone_model + '/test.txt')):
    if i == num_test_img: break
    d = line.split(',')
    img_id = int(d[0])
    img_ids.append(img_id)
    img_em = np.asarray(d[1:], dtype=np.float32)
    img_em = img_em / np.linalg.norm(img_em, 2)
    imgs_tensor[i,:] = img_em
print("Img embeddings loaded: " + str(len(img_ids)))
imgs_tensor = torch.from_numpy(imgs_tensor).cuda(gpu)


results = {}
with torch.no_grad():
    model_test.eval()
    for i, (tag, lat, lon, tag_str, lat_or, lon_or) in enumerate(test_loader):
        if i == num_query_pairs:
            print("Stopping at num tags: " + str(i))
            break
        st = time.time()
        scores = model_test(imgs_tensor, tag, lat, lon, gpu)
        scores = scores.squeeze(-1)
        top_values, top_img_indices = scores.topk(10)
        top_img_indices = np.array(top_img_indices.cpu()).tolist()
        top_img_ids = []
        for img_idx in top_img_indices:
            top_img_ids.append(img_ids[img_idx])
        key = tag_str[0] + ',' + str(lat_or[0]) + ',' + str(lon_or[0])
        results[key] = top_img_ids

        if i % 1 == 0:
            end = time.time()
            print(str(i) + ' / ' + str(num_query_pairs) + " Time per iter: " + str(end - st))

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")