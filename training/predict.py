import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

# called from main.py: predict.predict(model, args.predict, args.predict_outfile, SRC, TRG, logger)
def predict(model, infile, outfile, SRC, TRG, logger):
    model.eval()
    with open(infile, 'r') as in_f, open(outfile, 'w') as out_f:
        for i, line in enumerate(in_f):
            print(line)
            sent_german = line.split(' ') # next turn sentence into ints 
            sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sent_german]
            sent = Variable(torch.LongTensor([sent_indices]), volatile=True)
            if use_gpu: sent = sent.cuda()
            sent = sent.view(-1,1) # reshape to sl x bs
            # Predict with beam search 
            preds = model.predict_100(sent, beam_size=5) # predicts list of 100 lists of size 3
            final_preds = '' # string of the form word1|
            for pred in preds: # pred is list of size 3
                final_preds = final_preds + '{}|{}|{} '.format(preds[0], preds[1], preds[2])
            final_preds = final_preds.replace("\"", "<quote>").replace(",", "<comma>") # for Kaggle
            print(final_preds, file=out_f) # add to output file
            if i % 10 == 0: # log first 100 chars of each 10th prediction
                logger.log('German: {}\nEnglish: {}\n'.format(sent_german, final_preds[0:100]))
        logger.log('Finished predicting')
    return 
