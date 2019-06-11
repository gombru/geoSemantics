# Get the embeddings of all the test images

import os
import torch.utils.data
import model
import json
import numpy as np
import YFCC_dataset

# THIS CODE IS NOT READY

dataset_folder = '../../../hd/datasets/YFCC100M/'
test_im_dir = '../../../datasets/YFCC100M/test_img/'
split = 'test.txt'

batch_size = 600
workers = 6
ImgSize = 224

model_name = 'YFCC_MCLL_epoch_6_ValLoss_7.76.pth'
model_name = model_name.strip('.pth')

gpus = [1]
gpu = 1
CUDA_VISIBLE_DEVICES = 1

#
# def accuracy(output, target, topk=(1,)):
#
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res



if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/images_test.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0':'cuda:1', 'cuda:2':'cuda:1', 'cuda:3':'cuda:1'})


model_test = model.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset.YFCC_Dataset(test_im_dir, split, central_crop=ImgSize)
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