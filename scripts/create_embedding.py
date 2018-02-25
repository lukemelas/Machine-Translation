import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from utils.preprocess import preprocess

SRC, TRG, train_iter, val_iter = preprocess(0, 64) 
print('Data loaded.')

def load_embeddings(path, TEXT, embedding_dim=300):
    """ Creates a embedding from a file containing words and vector indices separated by spaces. Modified from https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb """
    with open(path) as f:
        embeddings = np.zeros((len(TEXT.vocab), embedding_dim))
        for i, line in enumerate(f.readlines()):
            values = line.split()
            word = values[0]
            if word in TEXT.vocab.stoi:
                index = TEXT.vocab.stoi[word]
                try:
                    vector = np.array(values[1:], dtype='float32')
                except:
                    vector = np.array([0] * 300, dtype='float32')
                    print('error: ', word)
                embeddings[index] = vector
            if i % 10000 == 0:
                print('{i} complete'.format(i=i)
        return embeddings 

# Save German embeddings
emb_de = load_embeddings('scripts/wiki.de.vec', SRC)
np.save('emb-{}-de'.format(str(len(SRC.vocab))), emb_de)
print('German embedding saved as np file')

# Save English embeddings
TRG.vocab.load_vectors('fasttext.simple.300d')
np.save('emb-{}-en'.format(len(TRG.vocab)), TRG.vocab.vectors.numpy())
print('English embedding saved as np file')

# Print sizes
print('German: {} \t English: {}'.format(emb_de.shape, TRG.vocab.vectors.numpy().shape))
