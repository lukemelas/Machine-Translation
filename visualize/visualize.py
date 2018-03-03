import os
os.environ['QT_QPA_PLATFORM']='offscreen' # for Azure
import pickle
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

def visualize(train_iter, model, SRC, TRG, logger):
    logger.log('VISUALIZING ATTENTION DISTRIBUTION')
    model.eval()
    for i, batch in enumerate(train_iter):
        if i in range (2,4): # which minibatch to visualize
            # Get prediction and attention distribution
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg
            scores, attn_dist = model.get_attn_dist(src, trg)
            # Loop over items in minibatch
            for k in range(src.size(1)): # src.size(1) = bs (= 64)
                src_bs1 = src.select(1,k).unsqueeze(1) # bs1 means batch size 1
                trg_bs1 = trg.select(1,k).unsqueeze(1) 
                attn_dist_bs1 = attn_dist.select(0,k)
                model.eval() # predict mode
                predictions = model.predict(src_bs1, beam_size=1)
                predictions_beam = model.predict(src_bs1, beam_size=2)
                model.train() # test mode
                probs, maxwords = torch.max(scores.data.select(1,k), dim=1) # training mode
                # Create lists
                src_list = list(SRC.vocab.itos[x] for x in src_bs1.squeeze().data)
                trg_list = list(TRG.vocab.itos[x] for x in trg_bs1.squeeze().data)  
                train_pred_list = list(TRG.vocab.itos[x] for x in maxwords) 
                greed_pred_list = list(TRG.vocab.itos[x] for x in predictions)
                beamy_pred_list = list(TRG.vocab.itos[x] for x in predictions_beam) 
                # Print lists
                logger.log('Source: ' + ' '.join(src_list))
                logger.log('Target: ' + ' '.join(trg_list))
                logger.log('Training Pred: ' + ' '.join(train_pred_list))
                logger.log('Validation Greedy Pred: ' + ' '.join(greed_pred_list))
                logger.log('Validation Beam Pred: ' + ' '.join(beamy_pred_list))
                logger.log('')
                # Display attn dist
                display_visual(src_list, train_pred_list, attn_dist_bs1, file_ext=str(k))
            return

def display_visual(src_list, trg_list, attn_dist, file_ext=''):
    # Remove <pad>, <s>, </s> tokens
    if '<pad>' in src_list:
        src_list = src_list[:src_list.index('<pad>')]
    if '<s>' in trg_list and '</s>' in trg_list:
        trg_list = trg_list[1:trg_list.index('</s>')]
    attn_dist = attn_dist[:len(src_list),:len(trg_list)]

    # From PyTorch seq2seq tutorial
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_dist.data.cpu().numpy(), cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + src_list, rotation=90) # axes
    ax.set_yticklabels([''] + trg_list) # axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # labels
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) # labels 
    plt.savefig('visualize/visualizations/attn' + file_ext + '.jpg')
    #plt.show()

