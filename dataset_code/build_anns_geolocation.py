import urllib
import json

# Reads the places metadata from YFCC100M
# Selects only the images with places metadata (48M)
# Use flickr API to get the GPS coordinates of the towns.
# Stores the image and location information in a txt file in the following format:
# id;[tag1,tag2,...];country;town;latitude;longitude;url


out_file = open("../../../ssd2/YFCC100M/anns_geo_gombru.txt",'w')

key = "71542c171848da257b3caa214f7ed00f"

print("Getting places metadata")
places_file = open("../../../ssd2/YFCC100M/yfcc100m_places")
places_metadata = {}
num_countries = 0
num_towns = 0

geo_info = {}

c = 0
for line in places_file:
    c+=1
    if c%200000 == 0: print(c)
    # if c == 10: break
    metadata = line.split(',')
    if len(metadata) < 2:
        continue  # No geolocation info
    country = ""
    town = ""
    town_id = 0

    id = int(metadata[0].split('\t')[0])
    metadata[0] = metadata[0].split('\t')[1]
    for field in metadata:
        if "Country" in field:
            country = field.split(':')[1]
            country_id = int(field.split(':')[0])
        elif "Town" in field:
            town = field.split(':')[1]
            town_id = int(field.split(':')[0])

    if country != "": num_countries+=1
    if town != "": num_towns+=1

    if town_id == 0: continue

    # Query GPS coords to Flickr
    if town_id not in geo_info:
        try:
            print(places_metadata[id]['town'])
            query_api_url = "https://api.flickr.com/services/rest/?method=flickr.places.getInfo&api_key="+key+"&woe_id="+str(town_id)+"&format=json&nojsoncallback=1"
            response = urllib.urlopen(query_api_url)
            data = json.loads(response.read())
            latitude = float(data['place']['latitude'])
            longitude = float(data['place']['longitude'])
            geo_info[town_id] = {}
            geo_info[town_id]['lat'] = str(latitude)
            geo_info[town_id]['lon'] = str(longitude)
        except:
            print("Error getting geo info for town. Skipping sample")
            continue

    # Has geolocation info
    places_metadata[id] = {}
    places_metadata[id]['town'] = town.replace(';',',')
    places_metadata[id]['town_id'] = town_id
    places_metadata[id]['country'] = country.replace(';',',')


places_file.close()

print("Number of elements with geolocation found: " + str(len(places_metadata)))
print("Number of elements with country found: " + str(num_countries))
print("Number of elements with town found: " + str(num_towns))


print("Getting images metadata")
dataset_file = open("../../../ssd2/YFCC100M/yfcc100m_dataset")
selected=0

c = 0
for line in dataset_file:
    c+=1
    if c%2000000 == 0: print(c)
    # if c == 100000: break
    metadata = line.split('\t')
    # print(metadata)
    id = int(metadata[1])
    if id not in places_metadata:
        continue  # No geolocation info
    type = metadata[-2]
    if type != 'jpg' and type != 'jpeg' and type != 'png':
        continue  # Is video
    url = metadata[-9].replace(';',',')
    tags = metadata[10].replace(';',',')
    if len(tags) < 3: continue # No tags

    # Image selected: Has geolocation info and tags
    selected+=1
    out_line = str(id) + ';' + tags + ';' + places_metadata[id]['country'] + ';' + places_metadata[id]['town'] + ';' + geo_info[town_id]['lat'] + ';'+ geo_info[town_id]['lon'] + ';' + url + '\n'
    # print(out_line)
    out_file.write(out_line)

print("Selected number of images: " + str(selected))

out_file.close()

print("DONE")