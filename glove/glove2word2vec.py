from gensim.scripts.glove2word2vec import glove2word2vec

in_path = '../../../ssd2/YFCC100M/text_models/glove.840B.300d.txt'
out_path = '../../../ssd2/YFCC100M/text_models/gensim_glove840B300d_vectors.txt'

glove2word2vec(glove_input_file=in_path, word2vec_output_file=out_path)