import itertools, os, sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .valid import validate
from utils.utils import AverageMeter

def train(train_iter, val_iter, model, criterion, optimizer, scheduler, SRC, TRG, num_epochs, logger=None):  

    # Iterate through epochs
    bleu_best = -1
    for epoch in range(num_epochs):
    
        # Step learning rate scheduler
        scheduler.step()

        # Validate model
        val_freq = 3
        if epoch % val_freq == 0: # and False: # skip for debug
            bleu_val = validate(val_iter, model, criterion, SRC, TRG, logger)
            #logger.log('Validation complete. BLEU: {:.3f}'.format(bleu_val))
            if bleu_val > bleu_best:
                bleu_best = bleu_val
                #logger.save_model(model.state_dict())
                logger.log('New best: {:.3f}'.format(bleu_best))

        # Train model
        losses = AverageMeter()
        model.train()
        for i, batch in enumerate(train_iter): 
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg
            
            # Forward, backprop, optimizer
            model.zero_grad()
            scores = model(src, trg)

            # Debug -- print sentences
            debug_print_sentences = False
            if i is 0 and debug_print_sentences:
                model.eval()
                predictions = model.predict(src)
                predictions_beam = model.predict_beam(src, TRG=TRG) # TRG for debug
                model.train()
                for j in range(4,7): #src.size(1)): # batch size
                    print('Source: ', ' '.join(SRC.vocab.itos[x] for x in batch.src.data.select(1,j)))
                    print('Target: ', ' '.join(TRG.vocab.itos[x] for x in batch.trg.data.select(1,j)))
                    probs, maxwords = torch.max(scores.data.select(1,j), dim=1)
                    print('Training Prediction: ', ' '.join(TRG.vocab.itos[x] for x in maxwords))
                    print('Validation Prediction: ', ' '.join(TRG.vocab.itos[x] for x in predictions[j]))
                    print('Validation Prediction: ', ' '.join(TRG.vocab.itos[x] for x in predictions_beam[j])) 
                    print()
                return

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
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

            # Log within epoch
            if i % 1000 == 999:
                logger.log('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, num_b=len(train_iter), l=losses.avg))

        # Log after each epoch
        logger.log('''Epoch [{e}/{num_e}] complete. Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, l=losses.avg))
        
        # DEBUG
        if epoch % 3 == 2:
            torch.save(model.state_dict(), 'saves/model.pkl')
