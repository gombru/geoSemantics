import json

loc_file = "../../../hd/datasets/YFCC100M/anns/img_geolocations.json"
splits_root = "../../../datasets/YFCC100M/splits/"
splits_out = "../../../datasets/YFCC100M/corrected_splits/"
splits = ['val','test','train_filtered']


print("Loading location data")
loc_data = json.load(open(loc_file))

for split in splits:
    print(split)
    not_found = 0
    out_file_split = open(splits_out + split + '.txt', 'w')

    for line in open(splits_root + split + '.txt'):
        d = line.split(';')
        id = d[0]
        try:
            d[4] = str(loc_data[id][0])
            d[5] = str(loc_data[id][1])
        except:
            # print("Not found")
            not_found+=1
            continue

        sep = ';'
        out_str = sep.join(d)
        # print(out_str)
        out_file_split.write(out_str)

    print("Not found in " + split + ": " + str(not_found))

print("DONE")



