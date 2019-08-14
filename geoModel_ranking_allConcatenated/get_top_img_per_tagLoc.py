# Get top images for tag-location pairs
# I will generate the query tag-location pairs before hand
# I will create a pair per test image (total 500k), chossing a random image tag and image location

import os
import torch.utils.data
import model
import json
import numpy as np
import YFCC_dataset_test_retrieval
import random
import time

random.seed(0)

dataset_folder = '../../../hd/datasets/YFCC100M/'
split = 'test.txt'
img_backbone_model = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55'

batch_size = 1
workers = 0
ImgSize = 224

num_query_pairs = 100000 # 500000
print("Using num query paris: " + str(num_query_pairs))

model_name = 'geoModel_ranking_allConcatenated_randomTriplets6Neg_MCLL_GN_TAGIMGL2_EML2_lr0_005_withLoc_epoch_5_ValLoss_0.02.pth'
model_name = model_name.replace('.pth','')

gpus = [2]
gpu = 2

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/tagLoc_top_img.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0':'cuda:2', 'cuda:1':'cuda:2', 'cuda:3':'cuda:2'})


model_test = model.Model_Test_Retrieval()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test_retrieval.YFCC_Dataset(dataset_folder, img_backbone_model, split)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

# Load GenSim Word2Vec model
print("Loading textual model ...")
text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
text_model = json.load(open(text_model_path))
print("Vocabulary size: " + str(len(text_model)))
print("Normalizing vocab")
for k,v in text_model.items():
    v = np.asarray(v, dtype=np.float32)
    text_model[k] = v / np.linalg.norm(v,2)

query_tags_tensor = np.zeros([num_query_pairs,300],  dtype=np.float32)
latitudes_tensor = np.zeros([num_query_pairs,1],  dtype=np.float32)
longitudes_tensor = np.zeros([num_query_pairs,1],  dtype=np.float32)

top_img_per_tagLoc_scores = torch.zeros([num_query_pairs,10],  dtype=torch.float32).cuda(gpu) - 1000
top_img_per_tagLoc_indices = torch.zeros([num_query_pairs,10], dtype=torch.int64).cuda(gpu)

print("Loading tag|loc queries ...")
queries_file = dataset_folder + 'geosensitive_queries/queries.txt'
query_tags_names = []
query_lats_str = []
query_lons_str = []
for i,line in enumerate(open(queries_file, 'r')):
    if i == num_query_pairs:
        print("Stopping at num_queries: " + str(i))
        break
    d = line.split(',')
    query_tags_names.append(d[0])
    lat = (float(d[1]) + 90) / 180
    lon = (float(d[2]) + 180) / 360
    query_lats_str.append(d[1])
    query_lons_str.append(d[2].replace('\n',''))
    query_tags_tensor[i,:] = np.asarray(text_model[d[0]], dtype=np.float32)
    latitudes_tensor[i,:] = lat
    longitudes_tensor[i,:] = lon

print("Number of queries: " + str(len(query_lats_str)))

query_tags_tensor = torch.from_numpy(query_tags_tensor).cuda(gpu)
latitudes_tensor = torch.from_numpy(latitudes_tensor).cuda(gpu)
longitudes_tensor = torch.from_numpy(longitudes_tensor).cuda(gpu)

with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):
        st = time.time()
        image_var = torch.autograd.Variable(image)
        scores = model_test(image_var, query_tags_tensor, latitudes_tensor, longitudes_tensor, gpu)
        scores = scores.squeeze(-1)
        values_to_replace, indices_to_replace = top_img_per_tagLoc_scores.min(dim=1)
        replacing_flags = scores > values_to_replace
        top_img_per_tagLoc_indices[replacing_flags, indices_to_replace[replacing_flags]] = float(img_id[0])
        top_img_per_tagLoc_scores[replacing_flags, indices_to_replace[replacing_flags]] = scores[replacing_flags]

        end = time.time()
        if i % 100 == 0 :
            print(str(i) + ' / ' + str(len(test_loader)) + " Time per iter: " + str(end-st))

print("Generating results")
results = {}
for i in range(0,num_query_pairs):
    key = query_tags_names[i] + ',' + str(query_lats_str[i]) + ',' + str(query_lons_str[i])
    results[key] = top_img_per_tagLoc_indices[i,:].cpu().detach().numpy().astype(int).tolist()

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")