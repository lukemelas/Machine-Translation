## LSTM for Language Modeling

This repository contains an implementation of a LSTM model for language modeling on the Penn Treebank database. More details on LSTM network architectures for state of the art language models may be found in [On the State of the Art of Evaluation in Neural Language Models](https://arxiv.org/abs/1707.05589).

To train our model, clone the repo and run `main.py`:
```
usage: main.py [-h] [--model DIR] [--lr N] [--hs N] [--nlayers N] [--no-wt]
               [--maxnorm N] [--dropout N] [-v N] [--data DATA] [-b N]                                                                       [--bptt N] [--epochs N] [--bigram] [-e] [-p] [--sample SAMPLE]
                                                                                                                              Language Model                                                                                                                
optional arguments:
  -h, --help       show this help message and exit
  --model DIR      path to model
  --lr N           learning rate
  --hs N           size of hidden state                                                                                         --nlayers N      number of layers in rnn                                                                                      --no-wt          disable weight tying in network
  --maxnorm N      maximum gradient norm for clipping
  --dropout N      dropout probability
  -v N             vocab size
  --data DATA      path to data
  -b N             batch size
  --bptt N         backprop though time length (sequence length)
  --epochs N       number of epochs
  --bigram         use bigram language model
  -e, --evaluate   run model only on validation set
  -p, --predict    save predictions on final input data
  --sample SAMPLE  number of sentences to sample
```

For example, we found the following hyperparameters worked well:

``` python main.py -b 128 --bptt 64 --epochs 20 --nlayers 2 ```



