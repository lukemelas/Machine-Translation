import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .Encoder import EncoderLSTM
from .Decoder import DecoderLSTM

class Seq2seq(nn.Module):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers, dropout_p, start_token_index=0):
        super(Seq2seq, self).__init__()
        self.h_dim = h_dim
        self.vocab_size = embedding_trg.size(0)
        self.start_token_index = start_token_index

        # Create encoder and decoder
        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p)
        self.decoder = DecoderLSTM(embedding_trg, h_dim, num_layers, dropout_p=dropout_p)

        # Create linear layers to combine context and hidden state
        self.linear1 = nn.Linear(2 * self.h_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.vocab_size)
        if False and self.decoder.embedding.weight.size().equals(linear2.weight.size()): # weight tying
            self.linear2.weight = self.decoder.embedding.weight

    def forward(self, src, trg):
        
        # Pass through encoder
        h_e_outs, h_e_final = self.encoder(src)

        # When training, process entire batch at once
        if self.training:

            # Pass through decoder with teacher forcing
            h_d_outs, h_d_final = self.decoder(trg, h_e_final) # sl x bs x h_dim, _

            # Generate context with attention
            context = h_d_final
            h_d_concat = torch.cat((h_d_final, context), dim=2) # sl x bs x 2 * h_dim

            # Pass through linear for probabilities
            scores = self.linear1(h_d_concat) # sl x bs x 2 * h_dim 
            scores = self.linear2(scores) # sl x bs x h_dim
            return scores

        else: # use beam search to find best sentence
            sents = []

            # Do sequentially so that batch size = 1
            for i in range(h_e_outs.size(1)): # loop over batches
            
                # Input hidden state is last state of encoder
                h_e_final0 = h_e_final[0][:,i,:].unsqueeze(1).contiguous()
                h_e_final1 = h_e_final[1][:,i,:].unsqueeze(1).contiguous()
                h_d_state = (h_e_final0, h_e_final1) # make bs = 1
                
                print('h_e_final[0].size(): ', h_e_final[0].size())
                print('h_e_final[1].size(): ', h_e_final[1].size())
                print('h_d_state[0].size(): ', h_d_state[0].size())
                print('h_d_state[1].size(): ', h_d_state[1].size())
                
                # Create initial starting word '<s>'
                word = torch.LongTensor([self.start_token_index] * h_d_state[0].size(1)).view(1, -1) # 1 x bs
                word = Variable(word, requires_grad=False)
                word = word.cuda() if use_gpu else word
                
                # Create list to hold our sentence
                sent = [self.start_token_index] # will store indices of words in sentence

                # Loop through sentence, ending when '</s>' or length 30 is reached
                for i in range(30): 
                
                    print('word: ', word, '\nsent: ', sent) 
                    
                    # Pass through decoder (batch size 1)
                    h_d_out, h_d_state = self.decoder(word, h_d_state)

                    print('h_d_out.size(): ', h_d_out.size())
                    print('h_d_state[0].size(): ', h_d_state[0].size()) 
                    print('h_d_state[1].size(): ', h_d_state[1].size()) 
                
                    # Use attention to compute context
                    context = h_d_out # 1 x bs x h_dim
                    h_d_concat = torch.cat((h_d_out, context), dim=2) # 1 x bs x 2 * h_dim

                    print('h_d_concat.size():, ', h_d_concat.size())
                    print('self.linear1:, ', self.linear1)

                    # Pass through linear for probabilities
                    scores = self.linear1(h_d_concat) # 1 x bs x h_dim
                    scores = self.linear2(scores) # 1 x bs x vs

                    print('scores.size(): ', scores.size())

                    probs = torch.nn.functional.softmax(scores) # 1 x bs x vs
                    ### !!! BE CAREFUL HERE IF INCREASING BATCH SIZE -- NEED TO SOFTMAX OVER VOCAB ONLY
                    print('probs.size(): ', probs.size())

                    # Get argmax for next word
                    prob, word = torch.max(probs, dim=2) # 1 x bs, 1 x bs
                    
                    word_index = word[0,0].data[0]
                    print('prob: ', prob, '\nword_index: ', word_index)
                    
                    # Set next word to word with highest prob
                    print('word: ', word, '\nword.requires_grad: ', word.requires_grad)

                    # Add word to current sentence
                    sent.append(word_index)

                    print('sent: ', sent)

                    # End translation if next word is '<s>'
                    if word_index is self.start_token_index or i is 29:
                        sents.append(sent)
                        i = 30 # break out of loop -- stop translating this sentence
            
            
            
            
            
            
            
            
            

    
# TODO
# - Implement beam search

