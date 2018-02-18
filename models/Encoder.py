import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class EncoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p

        # Create word embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p)

    def forward(self, x):

        # Embed text 
        x = self.embedding(x)

        # Create initial hidden state of zeros: 2 * nlayers x batch size x hidden dim
        h0 = Variable(torch.zeros(2 * self.nlayers, x.size(1), self.h_dim), requires_grad=False)

        # Pass through LSTM
        out, h = self.lstm(x, h0) # maybe have to pad now?
        return out, h

# Things to add:
# - dropout
# - bidirectional
