# Get the embeddings of all the test images

import os
import torch.utils.data
import model
import json
import numpy as np
import YFCC_dataset_test


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


if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/images_test.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0':'cuda:1', 'cuda:2':'cuda:1', 'cuda:3':'cuda:1'})


model_test = model.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test.YFCC_Dataset_Images_Test(test_im_dir, split, central_crop=ImgSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

top_img_per_tag = np.zeros((100000,10,2), dtype=int)

with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):
        image_var = torch.autograd.Variable(image)
        outputs = model_test(image_var)
        for idx,scores in enumerate(outputs):
            scores = np.array(scores.cpu()).tolist()
            for tag_idx, score in enumerate(scores):
                if score > min(top_img_per_tag[tag_idx,:,1]):
                    idx_to_replace = np.argmin(top_img_per_tag[tag_idx,:,1])
                    top_img_per_tag[tag_idx, idx_to_replace, 0] = str(img_id[idx])
                    top_img_per_tag[tag_idx, idx_to_replace, 1] = score
        print(str(i) + ' / ' + str(len(test_loader)))

results = {}
for i in range(0,100000):
    results[i] = top_img_per_tag[i,:,0].tolist()

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")