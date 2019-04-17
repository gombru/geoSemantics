

# DEPRECATED: Doesn't get Geolocation

# Reads the places metadata from YFCC100M
# Selects only the images with places metadata (48M)
# Stores the image and location information in a txt file in the following format:
# id;[tag1,tag2,...];country;town;url


out_file = open("anns_gombru.txt",'w')

print("Getting places metadata")
places_file = open("../../ssd2/YFCC100M/yfcc100m_places")
places_metadata = {}
num_countries = 0
num_towns = 0

c = 0
for line in places_file:
    c+=1
    if c%2000000 == 0: print(c)
    # if c == 100000: break
    metadata = line.split(',')
    if len(metadata) < 2:
        continue  # No geolocation info
    country = ""
    town = ""
    countries = []
    towns = []
    id = int(metadata[0].split('\t')[0])
    for field in metadata:
        if "Country" in field:
            countries.append(field.split(':')[1])
        elif "Town" in field:
            towns.append(field.split(':')[1])

    if len(countries) > 0:
        country = ','.join(countries)
        num_countries+=1

    if len(towns) > 0:
        town = ','.join(towns)
        num_towns+=1

    # Has geolocation info
    places_metadata[id] = {}
    places_metadata[id]['town'] = town.replace(';',',')
    places_metadata[id]['country'] = country.replace(';',',')


places_file.close()

print("Number of elements with geolocation found: " + str(len(places_metadata)))
print("Number of elements with country found: " + str(num_countries))
print("Number of elements with town found: " + str(num_towns))


print("Getting images metadata")
dataset_file = open("../../ssd2/YFCC100M/yfcc100m_dataset")
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
    out_line = str(id) + ';' + tags + ';' + places_metadata[id]['country'] + ';' + places_metadata[id]['town'] + ';' + url + '\n'
    out_file.write(out_line)

print("Selected number of images: " + str(selected))

out_file.close()

print("DONE")