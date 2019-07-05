# Get the embeddings of all the tags in vocabulary

import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_test
import YFCC_dataset_tags_test
import json
import numpy as np

dataset = '../../../hd/datasets/YFCC100M/'

batch_size = 5000
workers = 4
embedding_dims = 1024

model_name = 'YFCC_triplet_Img2Hash_e1024_m1_randomNeg_epoch_18_ValLoss_0.31.pth'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

if not os.path.exists(dataset + 'results/' + model_name):
    os.makedirs(dataset + 'results/' + model_name)

output_file_path = dataset + 'results/' + model_name + '/tags_embeddings.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:1', 'cuda:2':'cuda:1', 'cuda:3':'cuda:1'})


model_test = models_test.SuperTagsModel(embedding_dims=embedding_dims)
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_tags_test.YFCC_Dataset_Tags_Test()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True, sampler=None)


out_tag_embeddings = {}

with torch.no_grad():
    model_test.eval()
    for i, (tag, tag_text_model) in enumerate(test_loader):
        tag_text_model_var = torch.autograd.Variable(tag_text_model)
        outputs = model_test(tag_text_model_var)
        for idx,embedding in enumerate(outputs):
            out_tag_embeddings[str(tag[idx])] = np.array(embedding.cpu()).tolist()
        print(str(i) + ' / ' + str(len(test_loader)))

print("Writing results")

json.dump(out_tag_embeddings, output_file)
output_file.close()

print("DONE")
