import urllib
import cStringIO
from joblib import Parallel, delayed
from PIL import Image

def resize(im, minSize):
    w = im.size[0]
    h = im.size[1]
    if w < h:
        new_width = minSize
        new_height = int(minSize * (float(h) / w))
    if h <= w:
        new_height = minSize
        new_width = int(minSize * (float(w) / h))
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    return im

def download_save_image(id, url):
    img = cStringIO.StringIO(urllib.urlopen(url).read())
    img = resize(img, 300)
    image_path = dest_path + str(id) + '.jpg'
    img.save(image_path)

# Read anns
print("Reading anns")
ann_file = open("anns_gombru.txt")
dest_path = "home/Imatge/hd/datasets/YFCC100M/img/"
to_download = {}
for line in ann_file:
    data = line.split(';')
    id = data[0]
    url = data[-1]
    to_download[id] = url

print("Downloading")
Parallel(n_jobs=32)(delayed(download_save_image)(id, url) for id, url in to_download.iteritems())
print("DONE")