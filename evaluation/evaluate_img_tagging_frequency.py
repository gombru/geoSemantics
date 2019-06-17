# Image tagging evaluation using global frequency
# Measures Accuracy at K, typically k=1,10
# Accuracy at k: Is one of the K top predicted hashtags in the vocab?

import aux
import torch.nn as nn

dataset = '../../../hd/datasets/YFCC100M/'
test_split_path = '../../../datasets/YFCC100M/splits/test.txt'
ordered_vocab_path = '../../../datasets/YFCC100M/vocab/vocab_words_100k.txt'
accuracy_k = 10 # Compute accuracy at k (will also compute it at 1)

print("Reading tags of testing images ... ")
test_images_tags = aux.read_tags(test_split_path)

print("Reading frequency ordered vocab")
ordered_vocab = []
for line in open(ordered_vocab_path):
    ordered_vocab.append(line.strip('\n'))

print("Starting per-image evaluation")

total_accuracy_at_1 = 0.0
total_accuracy_at_k = 0.0

for i, (img_id, img_tags) in enumerate(test_images_tags.items()):

    if i % 500 == 0: print(i)
    img_id = str(img_id)

    # Compute Accuracy at 1
    if ordered_vocab[0] in img_tags:
        total_accuracy_at_1 += 1
    # Compute Accuracy at k
    for tag in ordered_vocab[0:accuracy_k]:
        if tag in img_tags:
            total_accuracy_at_k += 1
            break

total_accuracy_at_1 /= len(test_images_tags)
total_accuracy_at_k /= len(test_images_tags)

print("Accuracy at 1:" + str(total_accuracy_at_1))
print("Accuracy at " + str(accuracy_k) + " :" + str(total_accuracy_at_k))