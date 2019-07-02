import time

anns_file = open("../../../hd/datasets/YFCC100M/anns/anns_geo_filtered_gombru.txt", 'r')

for line in anns_file:
    print(line)
    time.sleep(5)


