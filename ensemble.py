import argparse, os, datetime, time, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext

from utils.utils import Logger, AverageMeter
from utils.preprocess import preprocess, load_embeddings
from models.Seq2seq import Seq2seq

# Different model files
model1file = 'saves/for-writeup/big-27-15-48-37/model.pkl'
model2file = 'saves/for-writeup/03-03-03-41-00/model.pkl'
model3file = 'saves/for-writeup/big-01-20-22-06/model.pkl'

# model1file = 'saves/mini-model.pkl'
# model2file = 'saves/mini-model-rv.pkl'
# model3file = 'saves/mini-model.pkl'

use_gpu = torch.cuda.is_available()

# Load data
time_data = time.time()
SRC, TRG, train_iter, val_iter = preprocess(0, 64)
print('Loaded data. |TRG| = {}. Time: {:.2f}.'.format(len(TRG.vocab), time.time() - time_data))

# Load embeddings
np_de_file = 'scripts/emb-{}-de.npy'.format(len(SRC.vocab))
np_en_file = 'scripts/emb-{}-en.npy'.format(len(TRG.vocab))
embedding_src, embedding_trg = load_embeddings(SRC, TRG, np_de_file, np_en_file)
print('Loaded embedding vectors from np files')

# Create model 
tokens = [TRG.vocab.stoi[x] for x in ['<s>', '</s>', '<pad>', '<unk>']]
model1 = Seq2seq(embedding_src, embedding_trg, 300, 2, 0.25, True, 'dot-product', tokens_bos_eos_pad_unk=tokens, reverse_input=False)
model1.load_state_dict(torch.load(model1file))
model2 = Seq2seq(embedding_src, embedding_trg, 300, 2, 0.30, True, 'dot-product', tokens_bos_eos_pad_unk=tokens, reverse_input=True)
model2.load_state_dict(torch.load(model2file))
model3 = Seq2seq(embedding_src, embedding_trg, 300, 2, 0.35, True, 'dot-product', tokens_bos_eos_pad_unk=tokens, reverse_input=False) 
model3.load_state_dict(torch.load(model3file))

# model1 = Seq2seq(embedding_src, embedding_trg, 300, 2, 0.25, True, 'dot-product', tokens_bos_eos_pad_unk=tokens, reverse_input=False)
# model1.load_state_dict(torch.load(model1file))
# model2 = Seq2seq(embedding_src, embedding_trg, 300, 2, 0.25, True, 'dot-product', tokens_bos_eos_pad_unk=tokens, reverse_input=True)
# model2.load_state_dict(torch.load(model2file))
# model3 = Seq2seq(embedding_src, embedding_trg, 300, 2, 0.25, True, 'dot-product', tokens_bos_eos_pad_unk=tokens, reverse_input=False)
# model3.load_state_dict(torch.load(model3file))

# Eval
model1.cuda()
model2.cuda()
model3.cuda()
model1.eval()
model2.eval()
model3.eval()

# Write file
infile = 'scripts/source_test.txt'
outfile = 'scripts/outfile.txt'

# called from main.py: predict.predict(model, args.predict, args.predict_outfile, SRC, TRG, logger)
with open(infile, 'r') as in_f, open(outfile, 'w') as out_f:
    print('id,word', file=out_f) # for Kaggle
    for i, line in enumerate(in_f):

        sent_german = line.split(' ') # next turn sentence into ints 
        sent_german[-1] = sent_german[-1][:-1] # remove '\n' from last word
        sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sent_german]
        sent = Variable(torch.LongTensor([sent_indices]), volatile=True)
        if use_gpu: sent = sent.cuda()
        sent = sent.view(-1,1) # reshape to sl x bs

        # Predict with each model
        remove_tokens = [TRG.vocab.stoi['</s>'], TRG.vocab.stoi['<unk>']] # block predictions of <eos> and <unk>
        k = 100 # set to 100
        max_len = 3
        preds1 = model1.beam_search(sent, k, max_len=max_len, remove_tokens=remove_tokens)
        preds2 = model2.beam_search(sent, k, max_len=max_len, remove_tokens=remove_tokens)
        preds3 = model3.beam_search(sent, k, max_len=max_len, remove_tokens=remove_tokens)

        # Combine and sort predictions
        all_preds = preds1 + preds2 + preds3
        all_preds.sort(key = lambda x: x[0], reverse=True) # now preds is a sorted list of length 300
    
        # Create final preds
        preds = [] 
        j = 0
        while(len(preds) < k):
            if all_preds[j][1] not in preds:
                preds.append(all_preds[j][1])
            j += 1

        # Turn preds into words
        final_preds = '{},'.format(i+1) # string of the form id,word1|word2|word3 word1|word2|word3 ...
        for pred in preds: # pred is list of size 3
            pred = pred[1:] # remove '<s>' from start of sentence
            pred = [TRG.vocab.itos[index] for index in pred] # convert indices to strings
            pred = [word.replace("\"", "<quote>").replace(",", "<comma>") for word in pred] # for Kaggle
            if len(pred) != 3: print('TOO SHORT: ', pred); continue # should not occur; just in case
            final_preds = final_preds + '{p[0]}|{p[1]}|{p[2]} '.format(p=pred) # for Kaggle

        print(final_preds, file=out_f) # add to output file

        if i % 5 == 0: # log first 100 chars of each 10th prediction
            print('German: {}\nEnglish: {}\n'.format(sent_german, final_preds[0:100]))


