import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools, os 
use_gpu = torch.cuda.is_available()

import itertools, os, datetime

class Logger():  
  def __init__(self):
    '''Create new log file'''
    j = 0
    while os.path.exists('log-{}.log'.format(j)):
      j += 1
    self.fname = 'log-{}.log'.format(j)
    
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
  

