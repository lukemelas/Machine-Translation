import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class RNNLM(nn.Module):
  def __init__(self, embedding, hidden_size, num_layers, bptt_len, dropout):
    super(RNNLM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.bptt_len = bptt_len
    
    # Create word embedding and copy of word embeddings
    vocab_size, embedding_size = embedding.size()
    self.embedding       = nn.Embedding(vocab_size, embedding_size)
    self.embedding_fixed = nn.Embedding(vocab_size, embedding_size)
    self.embedding.weight.data.copy_(embedding)
    self.embedding_fixed.weight.data.copy_(embedding) 
    self.embedding_fixed.weight.requires_grad = False
    self.fc_embeddings = nn.Linear(embedding_size * 2, embedding_size)

    # Create LSTM and linear layers 
    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.linear.weight = self.embedding.weight 
    self.dropout = nn.Dropout(dropout) 

  def forward(self, x, h):
    
    # Embed text and pass through LSTM
    x1 = self.embedding(x)
    x2 = self.embedding_fixed(x)
    x = x1 + x2 
    #torch.cat((x1,x2), dim=2)
    #x = self.fc_embeddings(x)
    x = self.dropout(x)
    out, h = self.lstm(x, h)
    out = self.linear(out)
    return out, h
