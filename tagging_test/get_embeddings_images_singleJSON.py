# Get the embeddings of all the test images

import os
import torch
import YFCC_dataset_images_test
import torch.utils.data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import model
import json
import numpy as np

dataset_folder = '../../../datasets/YFCC100M/'
test_im_dir = '/home/raulgomez/projects/tagging_test/img/'
split = 'img_list_filtered.txt'

batch_size = 200
workers = 1
ImgSize = 224

model_name = 'YFCC_NCSL_2ndtraining_epoch_16_ValLoss_0.38.pth'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

if not os.path.exists('results/'):
    os.makedirs('results/')

output_file_path = 'results/images_embeddings.json'
output_file = open(output_file_path, "w")

print("Loading model")
state_dict = torch.load(dataset_folder + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = model.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_images_test.YFCC_Dataset_Images_Test(test_im_dir, split, central_crop=ImgSize)
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