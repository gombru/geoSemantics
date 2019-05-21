from gensim.models.keyedvectors import KeyedVectors

print("Loading model")

text_model_path = '../../../ssd2/YFCC100M/text_models/gensim_glove840B300d_vectors.txt'
model = KeyedVectors.load_word2vec_format(text_model_path, binary=False, unicode_errors='ignore')

print(model.most_similar(positive=['amusement']))
print(model.most_similar(positive=['tree']))
print(model.most_similar(positive=['singing']))
print(model.most_similar(positive=['sing']))
print(model.most_similar(positive=['Singing']))
print(model.most_similar(positive=['barcelona']))
print(model.most_similar(positive=['Malaga']))
print(model.most_similar(positive=['canon']))
print(model.most_similar(positive=['cantar']))
print(model.most_similar(positive=['coche']))
print(model.most_similar(positive=['bicicleta']))
print(model.most_similar(positive=['viaje']))
print(model.most_similar(positive=['playa']))









