import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse, os
use_gpu = torch.cuda.is_available()

from train import train_model
from utils import Logger
from rnn import RNNLM
from preprocess import preprocess

parser = argparse.ArgumentParser(description='Language Model Sampler')
parser.add_argument('model', metavar='DIR', help='path to model')
parser.add_argument('lr', default=2e-3, type=int, metavar='N', help='learning rate')
parser.add_argument('hs', default=100, type=int, metavar='N', help='size of hidden state')
parser.add_argument('nlayers', default=1, type=int, metavar='N', help='number of layers in rnn')
parser.add_argument('maxnorm', default=1.0, type=float, metavar='N', help='maximum gradient norm for clipping')
parser.add_argument('v', default=1000, type=int, metavar='N', help='vocab size')
parser.add_argument('data', default='./data', help='path to data')
parser.add_argument('b', default=10, type=int, metavar='N', help='batch size')
parser.add_argument('bptt', default=32, type=int, metavar='N', help='backprop though time length (sequence length)')
parser.add_argument('', default=32, type=int, metavar='N', help='backprop though time length (sequence length)')


def main():
  global args
  args = parser.parse_args()
  
  # Load and process data
  TEXT = preprocess(args.data, args.b, args.bptt)
  
  # Create and load model
  embedding = TEXT.vocab.vectors.clone()
  model = RNNLM(embedding, args.hidden_size, args.nlayers, args.bptt)
  model = model.cuda() if use_gpu else model
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
  
  # Create logger and log parameters
  logger = Logger()
  logger.log(logger.log('''model: {m}
    optimizer: {o}
    learning rate: {lr}
    hidden_size: {hs}
    num_layers: {nl}
    max_norm: {mn}
    '''.format(m=model, o=optimizer, lr=args.lr, hs=args.hs, nl=args.nlayers, mn=args.mn), stdout=False)
  
  # Train model
  train_model(train_iter, val_iter, model, criterion, max_norm=args.maxnorm, num_epochs=args.num_epochs, logger=logger)
  
  

  model = 
  if os.path.isfile(args.model):
    
    sent = ''
    for _ in range(args.n):
      sent = sample(args.model, max_sample_length=args.max_len)
    print(sent)
  else:
    raise Exception('Please input valid model file')
  

if __name__ == '__main__':
  main()
