import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class BigramModel(nn.Module):
  def __init__(self, train_iter, TEXT, alpha=0.5, beta=0.5):
    super(BigramModel, self).__init__()
    self.vocab_size = len(TEXT.vocab)
    self.num_layers = self.hidden_size = 0
    self.param = torch.nn.Parameter(torch.FloatTensor(1))
    
    # Set up unigrams, bigrams counters
    self.unigram_dict = torch.zeros(self.vocab_size)
    self.bigram_dict = {}
    self.trigram_dict = {}
        
    # Set up previous word storage
    train_iterator = iter(train_iter)
    word1_store = next(train_iterator).text.data    
    word2_store = torch.cat((word1_store[1:], next(train_iterator).text.data[0].view(1,-1)), 0)
    
    # Iterate over training words
    for i, b in enumerate(train_iterator):
      
      # Create and save words
      word1 = word1_store
      word2 = word2_store
      word3 = torch.cat((word2_store[1:], b.text.data[0].view(1,-1)), 0)
      word1_store = word2_store
      word2_store = b.text.data
      
      # Create unigrams, bigrams, trigrams from words
      unigrams = word1.view(-1).tolist()
      bigrams  = torch.stack((word1, word2)).view(-1, 2).tolist()
      trigrams = torch.stack((word1, word2, word3)).view(-1, 3).tolist()
      
      # Count unigrams, bigrams, trigrams
      eos = TEXT.vocab.stoi['<eos>']
      for w1 in unigrams:
        if w1 is not eos:
          self.unigram_dict[w1] += 1
      for (w1, w2) in bigrams:
        if eos in (w1, w2):
          continue
        elif w1 in self.bigram_dict:
          self.bigram_dict[w1][w2] += 1
        else:
          self.bigram_dict[w1] = torch.zeros(self.vocab_size)
          self.bigram_dict[w1][w2] += 1
      for (w1, w2, w3) in trigrams:
        if eos in (w1, w2, w3):
          continue
        elif (w1, w2) in self.trigram_dict:
          self.trigram_dict[(w1, w2)][w3] += 1
        else:
          self.trigram_dict[(w1, w2)] = torch.zeros(self.vocab_size)
          self.trigram_dict[(w1, w2)][w3] += 1

      # Log progress
      if i % 1000 is 0:
        print('Completed [{}/{}]'.format(i, len(train_iter)))
    
    # Normalize probability distributions 
    self.unigram_dict = self.unigram_dict / self.unigram_dict.sum()
    for i in range(self.vocab_size):
      if i in self.bigram_dict:
        self.bigram_dict[i] = self.bigram_dict[i] / (1e-10 + self.bigram_dict[i].sum())
      for j in range(self.vocab_size):
        if (i,j) in self.trigram_dict:
          self.trigram_dict[(i,j)] = self.trigram_dict[(i,j)] / (1e-10 + self.trigram_dict[(i,j)].sum())
          
    # Set initial hyperparameters
    self.set_hyperparameters(alpha, beta)
  
  def set_hyperparameters(self, alpha, beta):
    self.alpha = alpha
    self.beta = beta
  
  def forward(self, text, __):
    text = text.data

    # Unigram probabilities
    unigram_probs = self.unigram_dict.unsqueeze(0).unsqueeze(0).expand(text.size(0), text.size(1), self.vocab_size)

    # Bigram and trigram probabilities 
    bigram_probs  = torch.zeros(text.size(0), text.size(1), self.vocab_size)
    trigram_probs = torch.zeros(text.size(0), text.size(1), self.vocab_size)
    for i in range(text.size(0)):
      for j in range(text.size(1)):
        word = text[i][j]
        if word in self.bigram_dict:
          bigram_probs[i][j] = self.bigram_dict[word]
        if i > 1:
          words = (text[i-1][j], text[i][j])
          if words in self.trigram_dict:
            trigram_probs[i][j] = self.trigram_dict[words] 
    
    # Linear combination of bigram and unigram models
    probs = (1 - self.alpha - self.beta) * unigram_probs + self.alpha * bigram_probs + self.beta * trigram_probs
    probs = probs / (1e-10 + probs.sum(dim=2).view(probs.size(0), probs.size(1), 1))
    
    # Take log to undo CrossEntropyLoss function applied in validation
    probs = Variable(torch.log(1e-10 + probs), volatile=True)
    probs = probs.cuda() if use_gpu else probs
    
    return probs, None
