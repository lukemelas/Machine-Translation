import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .Encoder import EncoderLSTM
from .Decoder import DecoderLSTM

class Seq2seq(nn.Module):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers, dropout_p, start_token_index=0, eos_token_index=1):
        super(Seq2seq, self).__init__()
        self.h_dim = h_dim
        self.vocab_size_trg = embedding_trg.size(0)
        self.start_token_index = start_token_index
        self.eos_token_index = eos_token_index

        # Create encoder and decoder
        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p)
        self.decoder = DecoderLSTM(embedding_trg, h_dim, num_layers, dropout_p=dropout_p)

        # Create linear layers to combine context and hidden state
        self.linear1 = nn.Linear(2 * self.h_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.vocab_size_trg)

        # Weight tying
        if False and self.decoder.embedding.weight.size().equals(linear2.weight.size()): # weight tying
            self.linear2.weight = self.decoder.embedding.weight

    def attention(self, out_d, out_e):
        return out_d

    def forward(self, src, trg):
        
        # Pass through encoder
        out_e, final_e = self.encoder(src)
        
        # Pass through decoder
        out_d, final_d = self.decoder(trg, final_e)

        # Apply attention
        context = self.attention(out_d, out_e)
        out_cat = torch.cat((out_d, context), dim=2) 

        # Pass through linear layers to predict next word and return word probabilities
        x = self.linear1(out_cat)
        x = self.linear2(x)
        return x

    def predict(self, src, trg=None):

        # Store predictions: list of len bs of items of size sl x vs
        sents = []

        # Pass src through encoder
        out_e, final_e = self.encoder(src)

        # Loop through batches: for the inner code, bs = 1
        for i in range(out_e.size(1)):
            
            # Initial hidden state is last state of encoder
            state_d = tuple(x.select(1,i).unsqueeze(1).contiguous() for x in final_e)
            
            #h_e_final0 = h_e_final[0][:,i,:].unsqueeze(1).contiguous()
            #h_e_final1 = h_e_final[1][:,i,:].unsqueeze(1).contiguous()
            #h_d_state = (h_e_final0, h_e_final1) # make bs = 1
                
            #print('h_e_final[0].size(): ', h_e_final[0].size())
            #print('h_e_final[1].size(): ', h_e_final[1].size())
            #print('h_d_state[0].size(): ', h_d_state[0].size())
            #print('h_d_state[1].size(): ', h_d_state[1].size())
                
            # Create initial starting word '<s>'
            word = Variable(torch.LongTensor([self.start_token_index]).view(1,1), requires_grad=False)
            word = word.cuda() if use_gpu else word
            word_index = word[0,0].data[0]

            # Store generated words in current sentence
            sent = []

            # Generate words until end of sentence token or max length reached
            j = 0 # counter for position in sentence
            while(word_index != self.eos_token_index and j < 30): # max len = 30

                # Debug: replace with ground truth
                if trg is not None: word = trg[j,i].view(1,1)

                # Add word to current sentence
                sent.append(word_index)

                # Pass through decoder one word at a time
                out_d, state_d = self.decoder(word, state_d)

                # Apply attention
                context = self.attention(out_d, out_e)
                out_cat = torch.cat((out_d, context), dim=2) 

                # Pass through linear layers to predict next word and return word probabilities
                x = self.linear1(out_cat)
                x = self.linear2(x)
                
                # Sample word from distribution (by taking maximum)
                probs, word = x.topk(1)
                word = word.view(1,1)
                word_index = word[0,0].data[0]
                
                # Update word counter
                j += 1

            # Append generated sentence to full list of sentences
            sents.append(sent)

        # Return list of sentences (list of list of words)
        return sents

