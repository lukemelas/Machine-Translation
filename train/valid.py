import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools, os, datetime
use_gpu = torch.cuda.is_available()

def validate_model(val_iter, model, criterion, TEXT, logger=None):
    model.eval()
  
    # Iterate over words in validation batch. 
    losses = AverageMeter()
    for i, batch in enumerate(val_iter):
        text = batch.text.cuda() if use_gpu else batch.text 
        targets = batch.target.cuda() if use_gpu else batch.target

        # Forward, calculate loss
        outputs, states = model(text, states)
        outputs = outputs.contiguous().view(outputs.size(0) * outputs.size(1), outputs.size(2))
        targets = targets.view(outputs.size(0)) 

        # model will return best sentence from beam search
        # validate with BLEU, PPL?, LOSS?
    
        # Calculate losses
        loss = criterion(outputs, targets)
        losses.update(loss.data[0])
  
        # Log information after validation
        ppl = torch.exp(torch.FloatTensor([losses.avg]))[0]
  info = 'Validation complete. MAP: {map:.3f}, \t Loss: {loss:.3f}, \t Sorta-Perplexity: {perplexity:.3f}'.format(
      map=map_avg, loss=loss_avg, perplexity=ppl)
  logger.log(info) if logger is not None else print(info)
  return ppl
