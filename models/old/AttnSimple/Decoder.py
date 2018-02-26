import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class DecoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0):
        super(DecoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        
        # Create word embedding, LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding) 
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, x, h0):

        # Embed text and pass through GRU
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.lstm(x, h0)
        return out, h


