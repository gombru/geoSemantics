import json
from gensim.models.keyedvectors import KeyedVectors

in_file = open("../../../ssd2/YFCC100M/anns_geo_gombru.txt",'r')
out_vocab = open("../../../ssd2/YFCC100M/vocab_words_100k.txt",'w')
out_vocab_embeddings = open("../../../ssd2/YFCC100M/vocab_100k.json",'w')
out_file = open("../../../ssd2/YFCC100M/anns_geo_filtered_gombru.txt",'w')

def num_there(s):
    return any(i.isdigit() for i in s)

# First find top 100k most frequent tags, which will be our vocabulary
# Filter out tags not in text model vocab
# Remove 10 most frequent ones
# Remove numeric ones

print("Creating vocab")
print(" -- Loading text model")
text_model_path = '../../../ssd2/YFCC100M/text_models/gensim_glove840B300d_vectors.txt'
text_model = KeyedVectors.load_word2vec_format(text_model_path, binary=False, unicode_errors='ignore')
print("-- Reading anns")
all_vocab = {}
vocab_embeddings = {}
for line in in_file:
    d = line.split(';')
    tags = d[1].split(',')
    for t in tags:
        t = t.lower()
        if num_there(t): continue # Discard numeric tags
        if t in all_vocab: all_vocab[t] += 1 # Tags already in vocab
        else: # Tags still not in vocab
            # Check that tag is in text model vocab
            try:
                vocab_embeddings[t] = text_model[t]
                all_vocab[t] = 1
            except:
                print("Tag not found in text model: " + t)
                continue



print(" --  Total vocab length: " + str(all_vocab.__len__()))

print("Filtering dict")
all_vocab_sorted = sorted(all_vocab, key=all_vocab.get, reverse=True)
top_ten = all_vocab_sorted[0:10]
filtered_vocab = all_vocab_sorted[10:100010]
del all_vocab
del all_vocab_sorted

print(" -- Top ten tags that are ignored: ")
print(top_ten)
print(" -- Top next ten tags: ")
print(filtered_vocab[0:10])
print(" -- Filtered vocab length: " + str(len(filtered_vocab)))
print(" -- Saving vocab")
vocab_embeddings_filtered = {}
for w in filtered_vocab:
    out_vocab.write(w + '\n')
    vocab_embeddings_filtered[w] = vocab_embeddings[w]
out_vocab.close()

del vocab_embeddings
json.dump(vocab_embeddings_filtered, out_vocab_embeddings)
out_vocab_embeddings.close()
del vocab_embeddings_filtered

# Now loop again though samples
# Filter out hashtags not in vocab
# Filter out samples withou hashtags
# Remove images with more than 15 hashtags

print("Filtering samples and tags")
total = 0
selected = 0
for line in in_file:
    total+=1
    d = line.split(';')
    tags = d[1].split(',')
    selected_tags = []
    for t in tags:
        t = t.lower()
        if t in filtered_vocab:
            selected_tags.append(t)
    if len(selected_tags) > 0 and len(selected_tags) <= 15:
        selected_tags_str = selected_tags[0]
        for t_i in range(1,len(selected_tags)):
            selected_tags_str += ',' + selected_tags[t_i]
        line2write = d[0] + ';' + selected_tags_str + ';' + d[2] + ';' + d[3] + ';' + d[4] + ';'+ d[5] + ';' + d[6] + '\n'
        out_file.write(line2write)
        selected+=1

in_file.close()
out_file.close()

print("Selected " + str(selected) + " samples from a total of " + str(total))
print("DONE")