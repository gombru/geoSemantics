import numpy as np

def read_embeddings(path):
    embeddings = {}
    for line in open(path):
        d = line.split(',')
        embeddings[d[0]] = d[np.asarray(d[1:], dtype=np.float32)]

def read_tags(path):
    tags = {}
    for i, line in enumerate(open(path)):
        data = line.split(';')
        img_id = int(data[0])
        tags_array = data[1].split(',')
        tags[img_id] = tags_array
    return tags

def read_tags_and_locations(path):
    tags = {}
    latitudes = {}
    longitudes = {}
    for i, line in enumerate(open(path)):
        data = line.split(';')
        img_id = int(data[0])
        tags_array = data[1].split(',')
        tags[img_id] = tags_array
        latitudes[img_id] = float(data[4])
        longitudes[img_id] = float(data[4])

    return tags, latitudes, longitudes