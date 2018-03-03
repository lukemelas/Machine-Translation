import os, sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from utils.utils import moses_multi_bleu, AverageMeter

def validate(val_iter, model, criterion, SRC, TRG, logger):
    model.eval()
  
    # Iterate over words in validation batch. 
    bleu = AverageMeter()
    sents_out = [] # list of sentences from decoder
    sents_ref = [] # list of target sentences 
    for i, batch in enumerate(val_iter):
        # Use GPU
        src = batch.src.cuda() if use_gpu else batch.src
        trg = batch.trg.cuda() if use_gpu else batch.trg
        # Get model prediction (from beam search)
        out = model.predict(src, beam_size=1) # list of ints (word indices) from greedy search
        ref = list(trg.data.squeeze())
        # Prepare sentence for bleu script
        remove_tokens = [TRG.vocab.stoi['<pad>'], TRG.vocab.stoi['<s>'], TRG.vocab.stoi['</s>']] 
        out = [w for w in out if w not in remove_tokens]
        ref = [w for w in ref if w not in remove_tokens]
        sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
        sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
        sents_out.append(sent_out)
        sents_ref.append(sent_ref)
    # Run moses bleu script 
    bleu = moses_multi_bleu(sents_out, sents_ref) 
    # Log information after validation
    logger.log('Validation complete. BLEU: {bleu:.3f}'.format(bleu=bleu))
    return bleu

def validate_losses(val_iter, model, criterion, logger):
    '''Calculate losses by teacher forcing on the validation set'''
    model.eval()
    losses = AverageMeter()
    for i, batch in enumerate(val_iter): 
        src = batch.src.cuda() if use_gpu else batch.src
        trg = batch.trg.cuda() if use_gpu else batch.trg
        # Reverse src tensor
        inv_index = torch.arange(src.size(0)-1, -1, -1).long()
        src = src.index_select(0, inv_index)
        # Forward 
        scores = model(src, trg)
        scores = scores[:-1]
        trg = trg[1:]           
        # Reshape for loss function
        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
        trg = trg.view(scores.size(0))
        num_words = (trg != 0).float().sum()
        # Calculate loss
        loss = criterion(scores, trg) 
        losses.update(loss.data[0])
    logger.log('Average loss on validation: {:.3f}'.format(losses.avg))
    return losses.avg
    
