import math
import geopy.distance
import time
import numpy as np


def distance(lat, lng, lat0, lng0):
    deglen = 110.25
    x = lat - lat0
    y = (lng - lng0)*math.cos(lat0)
    return deglen*math.sqrt(x*x + y*y)


def distance2(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)

    distance_km = geopy.distance.vincenty(coords_1, coords_2).km
    print(distance_km)
    return distance_km

lat = 70
lng = 30
lat0=35
lng0=5

st = time.time()
a = distance(lat, lng, lat0, lng0)
print("time on other: " +str((time.time() - st)))
st = time.time()
b = distance2(lat, lng, lat0, lng0)
print("time on other: " +str((time.time() - st)))

print(b)

v = np.zeros((17000000))
v[18000] = 19

st = time.time()
ii = np.where(v==19)
print(ii)
print("time on other: " +str((time.time() - st)*325))