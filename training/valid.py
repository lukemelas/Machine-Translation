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
        src = batch.src.cuda() if use_gpu else batch.src
        trg = batch.trg.cuda() if use_gpu else batch.trg
        # Get model prediction (from beam search)
        out = model.predict_beam(src) # list of ints (word indices)
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
