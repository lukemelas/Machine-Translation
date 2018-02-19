import argparse, os, datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext

from training import train, valid, predict
from utils.utils import Logger, AverageMeter
from utils.preprocess import preprocess
from models.Seq2seq import Seq2seq

parser = argparse.ArgumentParser(description='Language Model')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default 2e-3')
parser.add_argument('--hs', default=128, type=int, metavar='N', help='size of hidden state, default 128')
parser.add_argument('--emb', default=128, type=int, metavar='N', help='embedding size, default 128')
parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn, default 2')
parser.add_argument('--dp', default=0.0, type=float, metavar='N', help='dropout probability, default 0.0')
parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default 0')
parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default 64')
parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of epochs, default 15')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default None')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='only evaluate model, default False')
parser.add_argument('--predict', metavar='DIR', default=None, help='directory with final input data for predictions, default None')
parser.add_argument('--predict_outfile', metavar='DIR', default='data/preds.txt', help='file to output final predictions, default "data/preds.txt"')
parser.set_defaults(evaluate = False)

def main():
    global args
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Load and process data
    SRC, TRG, train_iter, val_iter = preprocess(args.v, args.b)
    print('Loaded data. |TRG| = {}'.format(len(TRG.vocab)))
  
    # Create model # perhaps try pretrained: # SRC.vocab.vectors.clone()
    embedding_src = (torch.rand(len(SRC.vocab), args.emb) - 0.5) * 2
    embedding_trg = (torch.rand(len(TRG.vocab), args.emb) - 0.5) * 2
    model = Seq2seq(embedding_src, embedding_trg, args.hs, args.nlayers, args.dp, start_token_index=TRG.vocab.stoi['<s>']) 

    # Load pretrained model 
    if args.model is not None and os.path.isfile(args.model):
      model.load_state_dict(torch.load(args.model))
      print('Loaded pretrained model.')
    model = model.cuda() if use_gpu else model

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
  
    # Create directory for logs, create logger, log hyperparameters
    path = os.path.join('saves', datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    os.makedirs(path, exist_ok=True)
    logger = Logger(path)
    logger.log('ARGS: {}, MODEL, {}'.format(args, model), stdout=False)
    
    # Train, validate, or predict
    if args.predict is not None:
        predict.predict(model, args.predict, args.predict_outfile, SRC, TRG)
    elif args.evaluate:
        valid.validate(val_iter, model, criterion, SRC, TRG, logger)
    else:
        train.train(train_iter, val_iter, model, criterion, optimizer,scheduler, SRC, TRG, args.epochs, logger)
    return

if __name__ == '__main__':
    main()
