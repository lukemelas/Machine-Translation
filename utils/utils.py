import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe
import itertools, os 
use_gpu = torch.cuda.is_available()

class Logger():  
    '''Prints to a log file and to standard output''' 
    def __init__(self):
      j = 0
      while os.path.exists('saves/log-{}.log'.format(j)):
        j += 1
      self.fname = 'saves/log-{}.log'.format(j)
    
    def log(self, info, stdout=True):
      with open(self.fname, 'a') as f:
        print(info, file=f)
      if stdout:
        print(info)

def AverageMeter():
    '''Computes and stores the average and current value. 
       Taken from the PyTorch ImageNet tutorial'''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
   
    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def predict(model, datafile, TEXT, print_preds=False, fname='preds.txt'):
    raise NotImplementedError()
 ''' model.eval()
  
  with open(os.path.join(datafile, fname), 'w') as fout:
    print('id,word', file=fout) # header
    for i, line in enumerate(open(os.path.join(datafile, 'input.txt')), 1): # iterate over lines
      sentence = [TEXT.vocab.stoi[word] for word in line.split(' ')]
      sentence = Variable(torch.LongTensor(sentence).view(-1,1), volatile=True)
      sentence = sentence.cuda() if use_gpu else sentence
      
      # Prepare initial hidden state of zeros 
      num_layers, batch_size, hidden_size = model.num_layers, 1, model.hidden_size
      init = Variable(torch.zeros(num_layers, batch_size, hidden_size), requires_grad=False)
      init = init.cuda() if use_gpu else init
      states = (init, init.clone())

      # Run model and get predictions 
      outputs, states = model(sentence, states)
      
      # Get predictions after 10 words
      outputs = outputs[9].squeeze()
      
      # Remove '<eos>' from predictions
      outputs[TEXT.vocab.stoi['<eos>']] = -1000
      
      # Get top 20 predictions
      scores, preds = outputs.topk(20)
      preds = preds.data.tolist()
      
      # Save to output file
      print("{},{}".format(str(i), ' '.join([TEXT.vocab.itos[i] for i in preds])), file=fout)
      
      # For debugging: print our predictions 
      if print_preds and i < 10:
        print('{}  --> {}'.format(line, ', '.join([TEXT.vocab.itos[i] for i in preds])))
'''
