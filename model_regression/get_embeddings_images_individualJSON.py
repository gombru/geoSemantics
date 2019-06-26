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
test_im_dir = '../../../ssd2/YFCC100M/train_img/'
split = 'train_filtered.txt'

batch_size = 700
workers = 2
ImgSize = 224

model_name = 'YFCC_NCSL_2ndtraining_epoch_16_ValLoss_0.38.pth'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

output_folder = dataset_folder + 'img_embeddings/' + model_name + '/' + split.replace('.txt','') + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


state_dict = torch.load(dataset_folder + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:0':'cuda:1', 'cuda:2':'cuda:1', 'cuda:3':'cuda:1'})


model_test = model.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_images_test.YFCC_Dataset_Images_Test(test_im_dir, split, central_crop=ImgSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)
with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):

        if i < 2000:
            print(i)
            print(img_id[0])
            continue

        image_var = torch.autograd.Variable(image)
        outputs = model_test(image_var)

        for idx,embedding in enumerate(outputs):
            out_dict = {}
            out_dict[str(img_id[idx])] = np.array(embedding.cpu()).tolist()
            with open(output_folder + str(img_id[idx]) + '.json', 'w') as outfile:
                json.dump(out_dict, outfile)

        print(str(i) + ' / ' + str(len(test_loader)))
        print(img_id[0])

print("DONE")