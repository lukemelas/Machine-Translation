import argparse, os, datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext

from training import train, valid, predict
from utils.utils import Logger, AverageMeter
from utils.preprocess import preprocess, load_embeddings
from models.Seq2seq import Seq2seq

parser = argparse.ArgumentParser(description='Language Model')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
parser.add_argument('--hs', default=300, type=int, metavar='N', help='size of hidden state, default: 300')
parser.add_argument('--emb', default=300, type=int, metavar='N', help='embedding size, default: 300')
parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn, default: 2')
parser.add_argument('--dp', default=0.0, type=float, metavar='N', help='dropout probability, default: 0.0')
parser.add_argument('--unidir', dest='bi', action='store_false', help='use unidirectional encoder, default: bidirectional')
parser.add_argument('--attn', default='dot-product', type=str, metavar='STR', help='attention type: dot-product or additive, default: dot-product ')
parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default: 0')
parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of epochs, default: 15')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='only evaluate model, default: False')
parser.add_argument('--predict', metavar='DIR', default=None, help='directory with final input data for predictions, default: None')
parser.add_argument('--predict_outfile', metavar='DIR', default='data/preds.txt', help='file to output final predictions, default: "data/preds.txt"')
parser.set_defaults(evaluate=False, bi=True)

def main():
    global args
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Load and process data
    SRC, TRG, train_iter, val_iter = preprocess(args.v, args.b)
    print('Loaded data. |TRG| = {}'.format(len(TRG.vocab)))
    
    # Load embeddings if available
    LOAD_EMBEDDINGS = True
    if LOAD_EMBEDDINGS:
        np_de_file = 'scripts/emb-{}-de.npy'.format(len(SRC.vocab))
        np_en_file = 'scripts/emb-{}-en.npy'.format(len(TRG.vocab))
        embedding_src, embedding_trg = load_embeddings(SRC, TRG, np_de_file, np_en_file)
        print('Loaded embedding vectors from np files')
    else:
        embedding_src = (torch.rand(len(SRC.vocab), args.emb) - 0.5) * 2
        embedding_trg = (torch.rand(len(TRG.vocab), args.emb) - 0.5) * 2
        print('Initialized embedding vectors')

    # Create model # perhaps try pretrained: # SRC.vocab.vectors.clone()
    model = Seq2seq(embedding_src, embedding_trg, args.hs, args.nlayers, args.dp, args.bi, args.attn, start_token_index=TRG.vocab.stoi['<s>'], eos_token_index=TRG.vocab.stoi['</s>'], pad_token_index=TRG.vocab.stoi['<pad>']) 

    # Load pretrained model 
    if args.model is not None and os.path.isfile(args.model):
      model.load_state_dict(torch.load(args.model))
      print('Loaded pretrained model.')
    model = model.cuda() if use_gpu else model

    # Create weight to mask padding tokens for loss function
    weight = torch.ones(len(TRG.vocab))
    weight[TRG.vocab.stoi['<pad>']] = 0
    weight = weight.cuda() if use_gpu else weight

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight) 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, cooldown=6)
        #MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
  
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
