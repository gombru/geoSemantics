# Get the embeddings of all the test images

import os
import torch
import YFCC_dataset_queries_test
import torch.utils.data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import model_tags_test
import json
import numpy as np

dataset_folder = '../../../hd/datasets/YFCC100M/'
split = 'test.txt'

batch_size = 1024
workers = 0
embedding_dims = 1024

model_name = 'geoModel_retrieval_CNN_NCSL_frozen_randomTriplets_noLoc_M1_iter_15000_TrainLoss_0.36.pth'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/queries_embeddings.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = model_tags_test.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_queries_test.YFCC_Dataset(dataset_folder, split)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

out_img_embeddings = {}


with torch.no_grad():
    model_test.eval()
    for i, (tag_str, tag, lat, lon, lat_str, lon_str) in enumerate(test_loader):
        tag_var = torch.autograd.Variable(tag)
        lat_var = torch.autograd.Variable(lat * 0)
        lon_var = torch.autograd.Variable(lon * 0)

        outputs = model_test(tag_var,lat_var,lon_var)

        for idx,embedding in enumerate(outputs):
            key = str(tag_str[idx]) + ',' + str(lat_str[idx]) + ',' + str(lon_str[idx])
            out_img_embeddings[key] = np.array(embedding.cpu()).tolist()
        print(str(i) + ' / ' + str(len(test_loader)))

print("Writing results")
json.dump(out_img_embeddings, output_file)
output_file.close()

print("DONE")