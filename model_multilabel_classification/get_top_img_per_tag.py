# Get the embeddings of all the test imagess

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
workers = 3
ImgSize = 224

model_name = 'YFCC_ImageNet_MLC_2ndtraining_epoch_0_ValLoss_0.07.pth'
model_name = model_name.strip('.pth')

gpus = [1]
gpu = 1

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/tags_top_img.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0':'cuda:1', 'cuda:2':'cuda:1', 'cuda:3':'cuda:1'})


model_test = model.Model_Test()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=True)

test_dataset = YFCC_dataset_test.YFCC_Dataset_Images_Test(test_im_dir, split, central_crop=ImgSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

top_img_per_scores = torch.ones([100000,10],  dtype=torch.float32).cuda(gpu) * -1000
top_img_per_tag_indices = torch.zeros([100000,10], dtype=torch.int64).cuda(gpu)


with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):

        image_var = torch.autograd.Variable(image)
        outputs = model_test(image_var)

        for idx,scores in enumerate(outputs):
            values_to_replace, indices_to_replace = top_img_per_scores.min(dim=1)
            replacing_flags = scores > values_to_replace
            # if torch.sum(replacing_flags) > 1:
                # print(replacing_flags.shape)
                # print(scores[0:5])
                # print("Replacing: " + str(torch.sum(replacing_flags)))

            top_img_per_tag_indices[replacing_flags, indices_to_replace[replacing_flags]] = float(img_id[idx])
            top_img_per_scores[replacing_flags, indices_to_replace[replacing_flags]] = scores[replacing_flags]

        # if i == 6: break
        print(str(i) + ' / ' + str(len(test_loader)))


print("Generating results")
results = {}
for i in range(0,100000):
    results[i] = top_img_per_tag_indices[i,:].cpu().detach().numpy().astype(int).tolist()

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")