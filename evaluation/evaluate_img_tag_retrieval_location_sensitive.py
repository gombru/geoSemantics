# Image retrieval for every tag in vocabulary
# Only evaluate tag if at least K test images have it
# Measure Precision at K

# But this code does location-sensitive evaluation: A retrieval result is only correct if it's near the location query.
# We use an evalution procedure similar to the one used in IMG2GPS paper:
# Different granularities: (street level (1km), city (25km), region (200km), country (750km) and continent (2500km)).

import aux
import torch
import torch.nn.functional as F
import torch.nn as nn
import operator
import random
from shutil import copyfile
import os
import json
import numpy as np
import geopy.distance


granularities = [2500,750,200,25,1] # granularities in km

def check_location(lat1,lon1,lat2,lon2):
    coords_1 = (lat1,lon1)
    coords_2 = (lat2, lon2)
    distance_km = geopy.distance.vincenty(coords_1, coords_2).km
    results = np.zeros(len(granularities))
    for i,gr in enumerate(granularities):
        if distance_km <= gr:
            results[i] = 1
        else:
            break