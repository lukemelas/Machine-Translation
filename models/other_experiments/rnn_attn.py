import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class RNNLM(nn.Module):
  def __init__(self, embedding, hidden_size, num_layers, bptt_len, max_length=9, dropout_prob=0.2):
    super(RNNLM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.bptt_len = bptt_len
    self.dropout_prob = dropout_prob
    
    # Create word embedding
    vocab_size, embedding_size = embedding.size()
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.embedding.weight.data.copy_(embedding)

    # Attentional layers
    self.attn1 = nn.Linear(self.hidden_size * 2, max_length)
    self.attn2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
    
    # Create LSTM and linear layers
    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=self.dropout)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.linear.weight = self.embedding.weight
    
    # Additional layers
    self.dropout = nn.Dropout(self.dropout_prob)
    self.softmax = nn.Softmax(dim=1)
    self.relu = nn.ReLU()
 
  def forward(self, x, h, hs):
    
    # Embed text
    x = self.embedding(x)
    x = self.dropout(x)
    
    # Apply attention
    
    print(x.size())
    
    attn_alpha = self.softmax(self.attn1(torch.cat((x, h), 1)))
    
    print(attn_alpha.size())
    
    attn_state = torch.bmm(attn_alpha.unsqueeze(0), hs.unsqueeze(0))
    
    print(attn_state.size())
    
    y = torch.cat((x, attn_state), 1)
    
    print(y.size())
    
    y = self.relu(self.attn2(y))
    
    # Pass through LSTM and linear layer
    out, h = self.lstm(y, h)
    out = self.linear(out)
    
    return out, h
