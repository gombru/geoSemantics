# Create train, val and test splits

ann_file = open("../../../ssd2/YFCC100M/anns_geo_filtered_gombru.txt")

train_file = open("../../../ssd2/YFCC100M/splits/train.txt","w")
val_file = open("../../../ssd2/YFCC100M/splits/val.txt","w")
test_file = open("../../../ssd2/YFCC100M/splits/test.txt","w")

test_samples = 1000000
val_samples = 500000

samples = []

c=0
print("Reading anns")
for line in ann_file:
    c+=1
    if c%1000000 == 0: print(c)
    samples.append(line)
ann_file.close()


samples.shuffle()

print("Writing splits")
for i,s in enumerate(samples):
    if i < test_samples: test_file.write(s + '\n')
    elif i < (test_samples+val_samples): val_file.write(s + '\n')
    else: train_file.write(s + '\n')

print("DONE")

