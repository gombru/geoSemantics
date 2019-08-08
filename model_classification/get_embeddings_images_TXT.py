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

dataset_folder = '../../../hd/datasets/YFCC100M/'
test_im_dir = '../../../datasets/YFCC100M/val_img/'
split = 'val.txt'

batch_size = 700
workers = 4
ImgSize = 224

model_name = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55.pth'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

output_folder = dataset_folder + 'img_embeddings_single/' + model_name + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = open(output_folder + split, 'w')

state_dict = torch.load(dataset_folder + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda', 'cuda:3': 'cuda:0'})

model_test = model.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_images_test.YFCC_Dataset_Images_Test(test_im_dir, split, central_crop=ImgSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)
with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):

        image_var = torch.autograd.Variable(image)
        outputs = model_test(image_var)

        for idx, embedding in enumerate(outputs):
            img_em_str = ','.join(map(str, np.array(embedding.cpu()).tolist()))
            out_str = str(img_id[idx]) + ',' + img_em_str + '\n'
            output_file.write(out_str)

        print(str(i) + ' / ' + str(len(test_loader)))

print("DONE")