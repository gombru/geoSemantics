# Get top images for tag-location pairs
# I will generate the query tag-location pairs before hand
# I will create a pair per test image (total 500k), chossing a random image tag and image locationsss

import os
import torch.utils.data
import model
import json
import numpy as np
import YFCC_dataset_test_retrieval
import random

random.seed(0)

dataset_folder = '../../../datasets/YFCC100M/'
split = 'test.txt'
img_backbone_model = 'YFCC_NCSL_2ndtraining_epoch_16_ValLoss_0.38'

batch_size = 1
workers = 3
ImgSize = 224

model_name = 'geoModel_ranking_allConcatenated_randomTriplets6Neg_2ndtraining_epoch_56_ValLoss_0.03.pth'
model_name = model_name.replace('.pth','')

gpus = [0]
gpu = 0

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/tagLoc_top_img.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = model.Model_Test_Retrieval()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test_retrieval.YFCC_Dataset(dataset_folder, img_backbone_model, split)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

top_img_per_tagLoc_scores = torch.zeros([100000,10],  dtype=torch.float32).cuda(gpu) - 1000
top_img_per_tagLoc_indices = torch.zeros([100000,10], dtype=torch.int64).cuda(gpu)


# Load GenSim Word2Vec model
print("Loading textual model ...")
text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
text_model = json.load(open(text_model_path))
print("Vocabulary size: " + str(len(text_model)))
print("Normalizing vocab")
for k,v in text_model.items():
    v = np.asarray(v, dtype=np.float32)
    text_model[k] = v / np.linalg.norm(v,2)


print("Reading tags and locations ...")
tags_names = []
tags_tensor = np.zeros([100000,300],  dtype=np.float32)
latitudes_tensor = np.zeros([100000,1],  dtype=np.float32)
longitudes_tensor = np.zeros([100000,1],  dtype=np.float32)

for i,(k,v) in enumerate(text_model.items()):
    tags_tensor[i,:] = text_model[k]
    tags_names.append(k)

tags_tensor = torch.from_numpy(tags_tensor).cuda()
latitudes_tensor = torch.from_numpy(latitudes_tensor).cuda()
longitudes_tensor = torch.from_numpy(longitudes_tensor).cuda()

with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):

        image_var = torch.autograd.Variable(image)
        scores = model_test(image_var, tags_tensor, latitudes_tensor, longitudes_tensor)
        scores = scores.squeeze(-1)
        values_to_replace, indices_to_replace = top_img_per_tagLoc_scores.min(dim=1)
        replacing_flags = scores > values_to_replace
        top_img_per_tagLoc_indices[replacing_flags, indices_to_replace[replacing_flags]] = float(img_id[0])
        top_img_per_tagLoc_scores[replacing_flags, indices_to_replace[replacing_flags]] = scores[replacing_flags]

        print(str(i) + ' / ' + str(len(test_loader)))

print("Generating results")
results = {}
for i in range(0,100000):
    key = tags_names[i]
    results[key] = top_img_per_tagLoc_indices[i,:].cpu().detach().numpy().astype(int).tolist()

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")