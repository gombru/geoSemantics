import os

in_train_file = open("../../../datasets/YFCC100M/splits/train.txt","r")
out_train_file = open("../../../datasets/YFCC100M/splits/train_filtered.txt","w")

selected = 0
total = 0
for line in in_train_file:
    total += 1
    img_id = line.split(';')[0]
    if os.path.isfile("/home/Imatge/ssd2/YFCC100M/train_img/" + line.split(';')[0] + ".jpg"):
        out_train_file.write(line)
        selected += 1

print("Selected " + str(selected) + " lines out of " +str(total))