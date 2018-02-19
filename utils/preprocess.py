import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe
import spacy
import itertools, os
use_gpu = torch.cuda.is_available()

def preprocess(vocab_size, batchsize, max_sent_len=20):
    '''Loads data from text files into iterators'''

    # Load text tokenizers
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize(text, lang='en'):
        if lang is 'de':
            return [tok.text for tok in spacy_de.tokenizer(text)]
        elif lang is 'en':
            return [tok.text for tok in spacy_en.tokenizer(text)]
        else:
            raise Exception('Invalid language')

    # Add beginning-of-sentence and end-of-sentence tokens 
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=lambda x: tokenize(x, 'de'))
    EN = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD)

    # Create sentence pair dataset with max length 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), filter_pred = lambda x: max(len(vars(x)['src']), len(vars(x)['trg'])) <= max_sent_len)

    # Build vocabulary and convert text to indices
    # Convert words that appear fewer than 5 times to <unk>
    if vocab_size > 0:
        DE.build_vocab(train.src, min_freq=5, max_size=vocab_size)
        EN.build_vocab(train.trg, min_freq=5, max_size=vocab_size)
    else:
        DE.build_vocab(train.src, min_freq=5)
        EN.build_vocab(train.trg, min_freq=5)

    # Create iterators to process text in batches of approx. the same length
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=batchsize, device=-1, repeat=False, sort_key=lambda x: len(x.src))
    
    return DE, EN, train_iter, val_iter
