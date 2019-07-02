# For each image-location pair get its score with each tag, and save the top10

import os
import torch.utils.data
import model_test
import json
import numpy as np
import YFCC_dataset_test_tagging


dataset_folder = '../../../datasets/YFCC100M/'
split = 'test.txt'

batch_size = 1
workers = 3
ImgSize = 224

model_name = 'geoModel_to_test.pth.tar'
model_name = model_name.strip('.pth')

gpus = [0]
gpu = 0

if not os.path.exists(dataset_folder + 'results/' + model_name):
    os.makedirs(dataset_folder + 'results/' + model_name)

output_file_path = dataset_folder + 'results/' + model_name + '/images_test.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset_folder + '/models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model_test = model_test.Model()
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = YFCC_dataset_test_tagging.YFCC_Dataset_Images_Test(dataset_folder, split, central_crop=ImgSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

# Load GenSim Word2Vec model
print("Loading textual model ...")
text_model_path = '../../../datasets/YFCC100M/vocab/vocab_100k.json'
text_model = json.load(open(text_model_path))
print("Vocabulary size: " + str(len(text_model)))
print("Putting vocab in a tensor using ordered tag list")
tags_tensor = np.zeros((100000, 300), dtype=np.float32)
tags_file = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
for i,line in enumerate(open(tags_file)):
    tag = line.replace('\n', '').lower()
    tags_tensor[i,:] =  np.asarray(text_model[tag], dtype=np.float32)
tags_tensor = torch.autograd.Variable(torch.from_numpy(tags_tensor).cuda())
print("Tags tensor created")


print("Running model...")
results = {}
with torch.no_grad():
    model_test.eval()
    for i, (img_id, img, lat, lon) in enumerate(test_loader):
        img = torch.autograd.Variable(img)
        tag = torch.autograd.Variable(tag)
        lat = torch.autograd.Variable(lat)
        lon = torch.autograd.Variable(lon)
        # Try to run it with BS of 100k (vocab size). Else create here mini-batches and stack results
        outputs = model_test(img, tags_tensor, lat, lon)
        top_values, top_tag_indices = torch.max(outputs, 10)
        results[str(img_id)] = {}
        results[str(img_id)]['tags_indices'] = np.array(top_tag_indices.cpu()).tolist()
        results[str(img_id)]['tags_scores'] = np.array(top_values.cpu()).tolist()
        print(str(i) + ' / ' + str(len(test_loader)))

print("Writing results")
json.dump(results, output_file)
output_file.close()

print("DONE")