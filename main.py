import argparse, os, datetime, time, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext

from training import train, valid, predict
from visualize import visualize
from utils.utils import Logger, AverageMeter
from utils.preprocess import preprocess, load_embeddings
from models.Seq2seq import Seq2seq

parser = argparse.ArgumentParser(description='Machine Translation with Attention')
parser.add_argument('--lr', default=2e-3, type=float, metavar='N', help='learning rate, default: 2e-3')
parser.add_argument('--hs', default=300, type=int, metavar='N', help='size of hidden state, default: 300')
parser.add_argument('--emb', default=300, type=int, metavar='N', help='embedding size, default: 300')
parser.add_argument('--nlayers', default=2, type=int, metavar='N', help='number of layers in rnn, default: 2')
parser.add_argument('--dp', default=0.30, type=float, metavar='N', help='dropout probability, default: 0.30')
parser.add_argument('--unidir', dest='bi', action='store_false', help='use unidirectional encoder, default: bidirectional')
parser.add_argument('--attn', default='dot-product', type=str, metavar='STR', help='attention: dot-product, additive or none, default: dot-product ')
parser.add_argument('--reverse_input', dest='reverse_input', action='store_true', help='reverse input to encoder, default: False')
parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default: 0')
parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of epochs, default: 50')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='only evaluate model, default: False')
parser.add_argument('--visualize', dest='visualize', action='store_true', help='visualize model attention distribution')
parser.add_argument('--predict', metavar='DIR', default=None, help='directory with final input data for predictions, default: None')
parser.add_argument('--predict_outfile', metavar='DIR', default='data/preds.txt', help='file to output final predictions, default: "data/preds.txt"')
parser.add_argument('--predict_from_input', metavar='STR', default=None, help='German sentence to translate')
parser.set_defaults(evaluate=False, bi=True, reverse_input=False, visualize=False)

def main():
    global args
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Load and process data
    time_data = time.time()
    SRC, TRG, train_iter, val_iter = preprocess(args.v, args.b)
    print('Loaded data. |TRG| = {}. Time: {:.2f}.'.format(len(TRG.vocab), time.time() - time_data))
    
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

    # Create model 
    tokens = [TRG.vocab.stoi[x] for x in ['<s>', '</s>', '<pad>', '<unk>']]
    model = Seq2seq(embedding_src, embedding_trg, args.hs, args.nlayers, args.dp, args.bi, args.attn, tokens_bos_eos_pad_unk=tokens, reverse_input=args.reverse_input)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, factor=0.25, verbose=True, cooldown=6)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,13,16,19], gamma=0.5)
  
    # Create directory for logs, create logger, log hyperparameters
    path = os.path.join('saves', datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    os.makedirs(path, exist_ok=True)
    logger = Logger(path)
    logger.log('COMMAND ' + ' '.join(sys.argv), stdout=False)
    logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(args, optimizer, args.lr, vars(scheduler), model), stdout=False)
    
    # Train, validate, or predict
    start_time = time.time()
    if args.predict_from_input is not None:
        predict.predict_from_input(model, args.predict_from_input, SRC, TRG, logger)
    elif args.predict is not None:
        predict.predict(model, args.predict, args.predict_outfile, SRC, TRG, logger)
    elif args.visualize:
        visualize.visualize(train_iter, model, SRC, TRG, logger)
    elif args.evaluate:
        valid.validate(val_iter, model, criterion, SRC, TRG, logger)
    else:
        train.train(train_iter, val_iter, model, criterion, optimizer,scheduler, SRC, TRG, args.epochs, logger)
    logger.log('Finished in {}'.format(time.time() - start_time))
    return

if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()
