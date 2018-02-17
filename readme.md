## Machine Translation with PyTorch

This repository contains an implementation of a Seq2seq neural network model for machine translation. More details on sequence to sequence machine translation and hyperparameter tuning may be found in [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906).

This repository is a work in progress. 

<!--
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
  --ngram          use ngram language model
  -e, --evaluate   run model only on validation set
  -p, --predict    save predictions on final input data
  --sample SAMPLE  number of sentences to sample
```

For example, we found the following hyperparameters worked well:

``` python main.py -b 128 --bptt 64 --epochs 20 --nlayers 2 ```
-->


