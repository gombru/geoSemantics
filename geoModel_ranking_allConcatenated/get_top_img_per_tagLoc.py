# Get top images for tag-location pairs
# I will generate the query tag-location pairs before hand
# I will create a pair per test image (total 500k), chossing a random image tag and image location

import os
import torch.utils.data
import model_test_retrieval
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

num_query_pairs = 1000

model_name = 'geoModel_to_test'
model_name = model_name.replace('.pth','')

gpus = [0]
gpu = 0

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/tagLoc_top_img.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = model_test_retrieval.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test_retrieval.YFCC_Dataset(dataset_folder, img_backbone_model, split)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

top_img_per_tagLoc_scores = torch.zeros([num_query_pairs,10],  dtype=torch.float32).cuda(gpu)
top_img_per_tagLoc_indices = torch.zeros([num_query_pairs,10], dtype=torch.int64).cuda(gpu)

print("Generating query tag-location pairs")
# Load GenSim Word2Vec model
print("Loading textual model ...")
text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
text_model = json.load(open(text_model_path))
print("Reading tags and locations ...")
query_tags_names = []
query_tags_tensor = torch.zeros([num_query_pairs,300],  dtype=torch.float32)
latitudes_tensor = torch.zeros([num_query_pairs,1],  dtype=torch.float32)
longitudes_tensor = torch.zeros([num_query_pairs,1],  dtype=torch.float32)

for i, line in enumerate(open('../../../datasets/YFCC100M/splits/' + split)):
    if i % 100000 == 0 and i != 0: print(i)
    if i == 1000:
        print("Stopping at 1000")
        break
    data = line.split(';')
    tag = random.choice(data[1].split(','))
    query_tags_tensor[i,:] = np.asarray(text_model[tag], dtype=np.float32)
    query_tags_names.append(tag)

    lat = float(data[4])
    lon = float(data[5])
    # Coordinates normalization
    lat = (lat + 90) / 180
    lon = (lon + 180) / 360

    latitudes_tensor[i,:] = lat
    longitudes_tensor[i,:] = lon

query_tags_tensor.cuda()
latitudes_tensor.cuda()
longitudes_tensor.cuda()

with torch.no_grad():
    model_test.eval()
    for i, (img_id, image) in enumerate(test_loader):

        image_var = torch.autograd.Variable(image)
        outputs = model_test_retrieval(image_var, query_tags_tensor, latitudes_tensor, longitudes_tensor)

        for idx,scores in enumerate(outputs):
            values_to_replace, indices_to_replace = top_img_per_tagLoc_scores.min(dim=1)
            replacing_flags = scores > values_to_replace

            top_img_per_tagLoc_indices[replacing_flags, indices_to_replace[replacing_flags]] = float(img_id[idx])
            top_img_per_tagLoc_scores[replacing_flags, indices_to_replace[replacing_flags]] = scores[replacing_flags]

        print(str(i) + ' / ' + str(len(test_loader)))

print("Generating results")
results = {}
for i in range(0,100000):
    results[i] = top_img_per_tagLoc_indices[i,:].cpu().detach().numpy().astype(int).tolist()

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")