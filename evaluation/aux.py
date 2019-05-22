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