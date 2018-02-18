import os, sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from utils.utils import moses_multi_bleu

def validate(val_iter, model, criterion, SRC, TGT, logger):
    model.eval()
  
    # Iterate over words in validation batch. 
    #losses = AverageMeter()
    bleu = AverageMeter()
    for i, batch in enumerate(val_iter):
        src = batch.src.cuda() if use_gpu else batch.src
        trg = batch.trg.cuda() if use_gpu else batch.trg

        # Forward using beam search
        nothing = trg.copy().zero_() # no ground truth
        outputs,  = model(src, nothing) # use beam search

        # Model will return best sentence from beam search
        # Prepare sentences for moses multi-bleu script
        out_sentences = []
        ref_sentences = []
        for i in range(len(outputs)):
            out = outputs[i]
            ref = trg[i].data 
            out_sent = ' '.join(TRG.vocab.itos[j] for j in out)
            ref_sent = ' '.join(TRG.vocab.itos[j] for j in ref)
            out_sentences.append(out_sent)
            ref_sentences.append(ref_sent)
        
        # Run moses multi-bleu script
        batch_bleu = moses_multi_bleu(out_sentences, ref_sentences)
        bleu.update(batch_bleu)
    
        # Calculate losses (?) -- how ?
        #outputs = outputs.contiguous().view(outputs.size(0) * outputs.size(1), outputs.size(2))
        #targets = targets.view(outputs.size(0)) 
        #loss = criterion(outputs, targets)
        #losses.update(loss.data[0])
  
        # Log information after validation
        logger.log('Validation complete. BLEU: {bleu:.3f}'.format(bleu=bleu.avg))
	return bleu.avg
