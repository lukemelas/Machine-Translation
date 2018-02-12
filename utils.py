import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe
import itertools, os 
use_gpu = torch.cuda.is_available()

def preprocess(datafile, vocab_size, batch_size, bptt_len):
  '''Loads data from text files into iterators and builds word embeddings'''
  train_file = os.path.join(datafile, 'train.txt')
  valid_file = os.path.join(datafile, 'valid.txt')

  # Create datasets
  TEXT = torchtext.data.Field()
  train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".", 
    train=train_file, validation=valid_file, test=valid_file, text_field=TEXT)
  
  # Create vocab, iterator, and word embeddings
  TEXT.build_vocab(train, max_size=vocab_size)
  train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits( (train, val, test), batch_size=batch_size, device=-1, bptt_len=bptt_len, repeat=False)
  TEXT.vocab.load_vectors('fasttext.simple.300d')

  return TEXT, train_iter, val_iter

class Logger():  
  def __init__(self):
    '''Create new log file'''
    j = 0
    while os.path.exists('saves/log-{}.log'.format(j)):
      j += 1
    self.fname = 'saves/log-{}.log'.format(j)
    
  def log(self, info, stdout=True):
    '''Print to log file and standard output'''
    with open(self.fname, 'a') as f:
      print(info, file=f)
    if stdout:
      print(info)

def sample(model, max_sample_length=20):
  model.eval()
  
  # Initial states for LSTM
  batch_size = 1 #train_iter.batch_size
  num_layers, hidden_size = model.num_layers, model.hidden_size
  init = Variable(torch.zeros(num_layers, batch_size, hidden_size))
  init = init.cuda() if use_gpu else init
  state = (init, init.clone())
  
  # Select random first word from vocabulary
  random_index = torch.multinomial(torch.ones(len(TEXT.vocab)), num_samples=1).unsqueeze(1)
  word = Variable(random_index, volatile=True)
  word = word.cuda() if use_gpu else word
  
  # Sample words from model output distribution
  sentence = []
  for i in range(max_sample_length):
    outputs, state = model(word, state)
    distribution = torch.exp(outputs.squeeze().data)
    next_word = torch.multinomial(distribution, 1)[0] # sample
    word.data.fill_(next_word)
    sentence.append(word.data[0,0])
  
    # Stop sampling at end of sentence
    if word.data[0,0] is TEXT.vocab.stoi['<eos>']:
      break
  
  # Create and return sentence
  final = ' '.join([TEXT.vocab.itos[i] for i in sentence])
  return final
  

