from gensim.models.keyedvectors import KeyedVectors

model_path = '../../../ssd2/YMCC100M/text_models/GoogleNews-vectors-negative300.bin'

model = KeyedVectors.load_word2vec_format(model_path, binary=True)

print(model.most_similar(positive=['amusement']))
print(model.most_similar(positive=['tree']))
print(model.most_similar(positive=['singing']))
print(model.most_similar(positive=['sing']))
print(model.most_similar(positive=['Singing']))
print(model.most_similar(positive=['barcelona']))
print(model.most_similar(positive=['Malaga']))
print(model.most_similar(positive=['Málaga']))
print(model.most_similar(positive=['Canon']))
print(model.most_similar(positive=['canon']))
print(model.most_similar(positive=['nikond80']))
print(model.most_similar(positive=['Málagadasdasd']))





