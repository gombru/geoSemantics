# Get the embeddings of all the test images

import os
import torch
import YFCC_dataset_images_test
import torch.utils.data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import model_images_test
import json
import numpy as np

dataset_folder = '../../../hd/datasets/YFCC100M/'
images_dir = '../../../datasets/YFCC100M/test_img/'
split = 'test.txt'

batch_size = 600
workers = 0

model_name = 'geoModel_retrieval_CNN_NCSL_frozen_randomTriplets_noLoc_M1_iter_15000_TrainLoss_0.36' #
model_name = model_name.strip('.pth')

gpus = [1]
gpu = 1
CUDA_VISIBLE_DEVICES = 1

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/images_embeddings_test.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:2':'cuda:1', 'cuda:0':'cuda:1', 'cuda:3':'cuda:1'})


model_test = model_images_test.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_images_test.YFCC_Dataset(images_dir, split)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

out_img_embeddings = {}

with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):
        image_var = torch.autograd.Variable(image)
        outputs = model_test(image_var)

        for idx,embedding in enumerate(outputs):
            out_img_embeddings[str(img_id[idx])] = np.array(embedding.cpu()).tolist()
        print(str(i) + ' / ' + str(len(test_loader)))

print("Writing results")
json.dump(out_img_embeddings, output_file)
output_file.close()

print("DONE")