import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class DecoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0):
        super(DecoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding_size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        
        # Create word embedding with optional weight tying
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding) 
        
        # Create LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p)
    
    def forward(self, x, h0):
        
        # Embed text
        x = self.embedding(x)

        # Pass through LSTM
        out, h = self.lstm(x, h0)
        return out, h

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2 * self.num_layers, batch_size, self.h_dim), requires_grad=False)
# To do
# - implement three-way weight tying (TWWT)
