# Get the embeddings of all the test images

import os
import torch
import models_test
import YFCC_dataset_images_test

dataset = '../../../hd/datasets/YFCC100M/'
split = 'test.txt'

batch_size = 200
workers = 8
embedding_dims = 1024
ImgSize = 224

model_name = 'YFCC_triplet_Img2Hash_e1024_m1_randomNeg'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0

if not os.path.exists(dataset + 'results/' + model_name):
    os.makedirs(dataset + 'results/' + model_name)

output_file_path = dataset + 'results/' + model_name + '/images_test.txt'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset + '/models/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = models_test.ImagesModel(embedding_dims=embedding_dims)
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict)

test_dataset = YFCC_dataset_images_test.YFCC_Dataset_Images_Test(dataset, split, central_crop=ImgSize)

model_test = model_test.Model(embedding_dims=embedding_dims).cuda(gpu)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True, sampler=None)

with torch.no_grad():
    model_test.eval()
    for i, (id, image) in enumerate(test_loader):

        image_var = torch.autograd.Variable(image)
        outputs = model_test(image_var)

        for idx,embedding in enumerate(outputs):
            embedding_str = ''
            for v in embedding:
                embedding_str = embedding_str + ',' + str(float(v))
            output_file.write(str(id[idx]) + ',' + embedding_str + '\n')

        print(str(i) + ' / ' + str(len(test_loader)))

output_file.close()

print("DONE")