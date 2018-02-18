import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools, os, sys
sys.path.append('../')
use_gpu = torch.cuda.is_available()

import validate
from utils.utils import AverageMeter

def train(train_iter, val_iter, model, criterion, optimizer, scheduler, SRC, TRG, num_epochs, logger=None):  
    
    # Iterate through epochs
    ppl_best = 1000
    for epoch in range(num_epochs):
    
        # Step learning rate scheduler
        scheduler.step()

        # Validate model
        ppl_val = validate_model(val_iter, model, criterion, TEXT, logger)
        if ppl_val < ppl_best:
            logger.save(model.state_dict())
            ppl_best = ppl_val
            print('New best: {}'.format(best_ppl))
    
        # Train model
        losses = AverageMeter()
        model.train()
        for i, batch in enumerate(train_iter): 
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg

            # Forward, backprop, optimizer
            model.zero_grad()
            outputs, states = model(src, trg)
            outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
            targets = targets.view(outputs.size(0))
            loss = criterion(outputs, targets) 
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

        # Log information
        if i % 1000 == 10:
            logger.log('''Epoch [{epochs}/{num_epochs}]
                       Batch [{batch}/{num_batches}]
                       Loss: {losses.avg:.3f}
                       Perplexity: {ppl:.3f}
                       '''.format(epochs=epoch+1, num_epochs=num_epochs, batch=i, num_batches=len(train_iter), losses=losses, ppl=0.0)
