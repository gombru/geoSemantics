from shutil import copyfile
from joblib import Parallel, delayed

idx_file = open("../../../ssd2/YFCC100M/anns/anns_geo_filtered_gombru.txt","r")
img_folder = "../../../hd/datasets/YFCC100M/img/"
img_out_folder = "../../../ssd2/YFCC100M/img/"

# c=0
# for line in idx_file:
#     c+=1
#     if c % 100000 == 0: print(c)
#     img_name = line.split(';')[0] + '.jpg'
#     copyfile(img_folder + img_name, img_out_folder + img_name)
#
# print("DONE")

def copy(img_name):
    copyfile(img_folder + img_name, img_out_folder + img_name)

print("Reading anns")
filenames = []
for line in idx_file:
    filenames.append(line.split(';')[0] + '.jpg')

print("Coping")

parallelizer = Parallel(n_jobs=12)
tasks_iterator = (delayed(copy)(d) for d in filenames)

print("DONE")