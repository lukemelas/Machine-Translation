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
    
    # Create word embedding
    vocab_size, embedding_size = embedding.size()
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.embedding.weight.data.copy_(embedding)
    
    # Create LSTM and linear layers 
    self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.linear.weight = self.embedding.weight 
    self.dropout = nn.Dropout(dropout) 

  def forward(self, x, h):
    
    # Embed text and pass through LSTM
    x = self.embedding(x)
    x = self.dropout(x)
    if isinstance(h, tuple):
      h = h[0]
    if h.size(0) != 1:
      h = h.unsqueeze(0)

    out, hs = self.gru(x, h)
    
    # Reshape and pass through linear layer
    out = self.linear(out)
    return out, hs
