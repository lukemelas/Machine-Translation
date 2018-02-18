import argparse, os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext

from train import train, validate, predict
from utils.utils import Logger, AverageMeter
from utils.preprocess import preprocess
from models import Seq2seq

parser = argparse.ArgumentParser(description='Language Model')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate')
parser.add_argument('--hs', default=128, type=int, metavar='N', help='size of hidden state')
parser.add_argument('--emb', default=128, type=int, metavar='N', help='embedding size')
parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn')
parser.add_argument('--dp', default=0.0, type=float, metavar='N', help='dropout probability')
parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default 0')
parser.add_argument('-b', default=10, type=int, metavar='N', help='batch size')
parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of epochs')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='only evaluate model')
parser.add_argument('--predict', metavar='DIR', default=None, help='directory with final input data for predictions')
parser.add_argument('--predict_outfile', metavar='DIR', default='data/preds.txt', help='file to output final predictions')
parser.set_defaults(evaluate = False)

def main():
    global args
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Load and process data
    SRC, TRG, train_iter, val_iter = preprocess(args.v, args.b)
    print('Loaded data')
    return 
  
    # Create model # perhaps try pretrained: # SRC.vocab.vectors.clone()
    embedding_src = torch.FloatTensor(len(SRC.vocab), args.emb)
    embedding_trg = torch.FloatTensor(len(TRG.vocab), args.emb)
    model = Seq2seq(embedding_src, embedding_trg, args.hs, args.nlayers, args.dp) 

    # Load pretrained model 
    if args.model is not None and os.path.isfile(args.model):
      model.load_state_dict(torch.load(args.model))
      print('Loaded pretrained model.')
    model = model.cuda() if use_gpu else model

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) 
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
  
    # Create logger/saver and log hyperparameters
    logger = Logger()
    logger.log('ARGS: {}, MODEL, {}'.format(args, model), stdout=False)    
    
    # Train, validate, or predict
    if args.predict is not None:
        predict(model, args.predict, args.predict_outfile, SRC, TRG)
    elif args.evaluate:
        validate(val_iter, model, criterion, SRC, TRG, logger)
    else:
        train(train_iter, val_iter, model, criterion, optimizer,scheduler, SRC, TRG, num_epochs, logger)

  return

if __name__ == '__main__':
  main()
