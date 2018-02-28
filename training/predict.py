import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

# called from main.py: predict.predict(model, args.predict, args.predict_outfile, SRC, TRG, logger)
def predict(model, infile, outfile, SRC, TRG, logger):
    model.eval()
    with open(infile, 'r') as in_f, open(outfile, 'w') as out_f:
        print('id,word', file=out_f) # for Kaggle
        for i, line in enumerate(in_f):
            sent_german = line.split(' ') # next turn sentence into ints 
            sent_german[-1] = sent_german[-1][:-1] # remove '\n' from last word
            sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sent_german]
            sent = Variable(torch.LongTensor([sent_indices]), volatile=True)
            if use_gpu: sent = sent.cuda()
            sent = sent.view(-1,1) # reshape to sl x bs
            # Predict with beam search 
            final_preds = '{},'.format(i+1) # string of the form id,word1|word2|word3 word1|word2|word3 ...
            remove_tokens = [TRG.vocab.stoi['</s>'], TRG.vocab.stoi['<unk>']] # block predictions of <eos> and <unk>
            preds = model.predict_k(sent, 100, max_len=3, remove_tokens=remove_tokens) # predicts list of 100 lists of size 3
            for pred in preds: # pred is list of size 3
                pred = pred[1:] # remove '<s>' from start of sentence
                pred = [TRG.vocab.itos[index] for index in pred] # convert indices to strings
                pred = [word.replace("\"", "<quote>").replace(",", "<comma>") for word in pred] # for Kaggle
                if len(pred) != 3: print('TOO SHORT: ', pred); continue # should not occur; just in case
                final_preds = final_preds + '{p[0]}|{p[1]}|{p[2]} '.format(p=pred) # for Kaggle
            print(final_preds, file=out_f) # add to output file
            if i % 25 == 0: # log first 100 chars of each 10th prediction
                logger.log('German: {}\nEnglish: {}\n'.format(sent_german, final_preds[0:100]))
        logger.log('Finished predicting')
    return 

def predict_from_input(model, input_sentence, SRC, TRG, logger):
    sent_german = input_sentence.split(' ') # sentence --> list of words
    sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sent_german]
    sent = Variable(torch.LongTensor([sent_indices]), volatile=True) 
    if use_gpu: sent = sent.cuda()
    sent = sent.view(-1,1) # reshape to sl x bs
    logger.log('German: ' + ' '.join([SRC.vocab.itos[index] for index in sent_indices]))
    # Predict five sentences with beam search 
    preds = model.predict_k(sent, 5) # returns list of 5 lists of word indices
    for i, pred in enumerate(preds): # loop over top 5 sentences
        pred = [index for index in pred if index not in [TRG.vocab.stoi['<s>'], TRG.vocab.stoi['</s>']]]
        out = str(i+1) + ': ' + ' '.join([TRG.vocab.itos[index] for index in pred])
        logger.log(out)
    return 

