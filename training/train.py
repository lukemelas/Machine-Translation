import itertools, os, sys, time
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .valid import validate, validate_losses
from utils.utils import AverageMeter

def train(train_iter, val_iter, model, criterion, optimizer, scheduler, SRC, TRG, num_epochs, logger=None):  

    # Iterate through epochs
    bleu_best = -1
    for epoch in range(num_epochs):
    
        # Validate model with BLEU
        start_time = time.time() # timer 
        bleu_val = validate(val_iter, model, criterion, SRC, TRG, logger)
        if bleu_val > bleu_best:
            bleu_best = bleu_val
            logger.save_model(model.state_dict())
            logger.log('New best: {:.3f}'.format(bleu_best))
        val_time = time.time()
        logger.log('Validation time: {:.3f}'.format(val_time - start_time))

        # Validate model with teacher forcing (for PPL)
        val_loss = 0 #validate_losses(val_iter, model, criterion, logger) 
        logger.log('PPL: {:.3f}'.format(torch.FloatTensor([val_loss]).exp()[0])) 

        # Step learning rate scheduler
        scheduler.step(bleu_val) # input bleu score

        # Train model
        model.train()
        losses = AverageMeter()
        for i, batch in enumerate(train_iter): 
            # Use GPU
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg
            
            # Forward, backprop, optimizer
            model.zero_grad()
            scores = model(src, trg)

            # Debug -- print sentences
            debug_print_sentences = False
            if i is 0 and debug_print_sentences:
                for k in range(src.size(1)):
                    src_bs1 = src.select(1,k).unsqueeze(1) # bs1 means batch size 1
                    trg_bs1 = trg.select(1,k).unsqueeze(1) 
                    model.eval() # predict mode
                    predictions = model.predict(src_bs1, beam_size=1)
                    predictions_beam = model.predict(src_bs1, beam_size=2)
                    model.train() # test mode
                    probs, maxwords = torch.max(scores.data.select(1,k), dim=1) # training mode
                    logger.log('Source: ', ' '.join(SRC.vocab.itos[x] for x in src_bs1.squeeze().data))
                    logger.log('Target: ', ' '.join(TRG.vocab.itos[x] for x in trg_bs1.squeeze().data))
                    logger.log('Training Pred: ', ' '.join(TRG.vocab.itos[x] for x in maxwords))
                    logger.log('Validation Greedy Pred: ', ' '.join(TRG.vocab.itos[x] for x in predictions))
                    logger.log('Validation Beam Pred: ', ' '.join(TRG.vocab.itos[x] for x in predictions_beam)) 
                    logger.log()
                return # end after debugging

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            trg = trg[1:]           

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            trg = trg.view(scores.size(0))

            # Pass through loss function
            loss = criterion(scores, trg) 
            loss.backward()
            losses.update(loss.data[0])

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            # Log within epoch
            if i % 1000 == 10:
                logger.log('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, num_b=len(train_iter), l=losses.avg))

        # Log after each epoch
        logger.log('''Epoch [{e}/{num_e}] complete. Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, l=losses.avg))
        logger.log('Training time: {:.3f}'.format(time.time() - val_time))
        
