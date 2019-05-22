# Get the embeddings of all the tags in vocabulary

import os
import torch
import models_test
import YFCC_dataset_tags_test

dataset = '../../../hd/datasets/YFCC100M/'

batch_size = 512
workers = 4
embedding_dims = 1024

model_name = 'YFCC_triplet_Img2Hash_e1024_m1_randomNeg'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

if not os.path.exists(dataset + 'results/' + model_name):
    os.makedirs(dataset + 'results/' + model_name)

output_file_path = dataset + 'results/' + model_name + '/tags.txt'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = models_test.TagsModel(embedding_dims=embedding_dims)
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict)

test_dataset = YFCC_dataset_tags_test.YFCC_Dataset_Tags_Test()

model_test = model_test.Model(embedding_dims=embedding_dims).cuda(gpu)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True, sampler=None)

with torch.no_grad():
    model_test.eval()
    for i, (tag, tag_text_model) in enumerate(test_loader):

        tag_text_model_var = torch.autograd.Variable(tag_text_model)
        outputs = model_test(tag_text_model_var)

        for idx,embedding in enumerate(outputs):
            embedding_str = ''
            for v in embedding:
                embedding_str = embedding_str + ',' + str(float(v))
            output_file.write(str(tag[idx]) + ',' + embedding_str + '\n')

        print(str(i) + ' / ' + str(len(test_loader)))

output_file.close()

print("DONE")
