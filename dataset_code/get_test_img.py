from shutil import  copyfile

test_file = open("../../../ssd2/YFCC100M/splits/test.txt","r")
img_folder = "../../../hd/datasets/YFCC100M/img/"
img_out_folder = "../../../ssd2/YFCC100M/test_img/"

for line in test_file:
    img_name = line.split(';')[0] + '.jpg'
    copyfile(img_folder + img_name, img_out_folder + img_name)

print("DONE")

