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
        
        print('trg=word: ', trg)
        print('final_e=state_d: ', final_e[0][:,0,:])

        # Pass through decoder
        out_d, final_d = self.decoder(trg, final_e)

        # Apply attention
        context = self.attention(out_d, out_e)
        out_cat = torch.cat((out_d, context), dim=2) 

        # Pass through linear layers to predict next word and return word probabilities
        x = self.linear1(out_cat)
        x = self.linear2(x)
        return x

        
    def predict(self, src):

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
            #word = torch.LongTensor([self.start_token_index] * h_d_state[0].size(1)).view(1, -1) # 1 x bs
            #word = Variable(word, requires_grad=False)

            # Store generated words in current sentence
            sent = []

            if i is 0:
                print('word: ', word)
                print('state_d: ', state_d)
            
            # Generate words until end of sentence token or max length reached
            j = 0 # counter for position in sentence
            while(word_index != self.eos_token_index and j < 30): # max len = 30

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
                probs, word = torch.max(x, dim=2)
                word_index = word[0,0].data[0]
                
                # Update word counter
                j += 1

            # Append generated sentence to full list of sentences
            sents.append(sent)

            # Debug: print sent
            #print(sent)

        # Return list of sentences (list of list of words)
        return sents

# TODO

if False:
    print('''
                for j in range(trg.size(0)): # loop through words in sentence

                    # Set the next word to the next true word
                    word = trg[j,i].view(1,1) # 1 x bs = 1 x 1

                    print('word.size(): ', word.size())

                    # Pass through decoder
                    h_d_out, h_d_state = self.decoder(word, h_d_state)

                    # Use attention to compute context
                    context = h_d_out # 1 x bs x h_dim
                    h_d_concat = torch.cat((h_d_out, context), dim=2) # 1 x bs x 2 * h_dim

                    #print('h_d_concat.size():, ', h_d_concat.size())
                    #print('self.linear1:, ', self.linear1)

                    # Pass through linear for probabilities
                    scores = self.linear1(h_d_concat) # 1 x bs x h_dim
                    scores = self.linear2(scores) # 1 x bs x vs

                    print('scores.size(): ', scores.size())

                    # Add scores to list
                    preds.append(scores)

                # Concatenate predictions into single tensor
                preds = torch.cat(preds, dim=0)

                all_preds.append(preds)

           
            else: 
                # Create list to hold our sentence
                sent = [self.start_token_index] # will store indices of words in sentence

                # Loop through sentence, ending when '</s>' or length 30 is reached
                for j in range(30): 

                    print('WORD: ', word)
                
                    # Pass through decoder (batch size 1)
                    h_d_out, h_d_state = self.decoder(word, h_d_state)

                    if j > 3:
                        raise Exception()

                    #print('h_d_out.size(): ', h_d_out.size())
                    #print('h_d_state[0].size(): ', h_d_state[0].size()) 
                    #print('h_d_state[1].size(): ', h_d_state[1].size()) 
                
                    # Use attention to compute context
                    context = h_d_out # 1 x bs x h_dim
                    h_d_concat = torch.cat((h_d_out, context), dim=2) # 1 x bs x 2 * h_dim

                    #print('h_d_concat.size():, ', h_d_concat.size())
                    #print('self.linear1:, ', self.linear1)

                    # Pass through linear for probabilities
                    scores = self.linear1(h_d_concat) # 1 x bs x h_dim
                    scores = self.linear2(scores) # 1 x bs x vs

                    #print('scores.size(): ', scores.size())

                    ### !!! NEED UPGRADED PYTORCH -- NEED TO SOFTMAX OVER VOCAB ONLY
                    probs = scores # torch.nn.functional.softmax(scores, dim=2) # 1 x bs x vs
                    #print('probs.size(): ', probs.size())

                    #print('probs: ', probs)

                    # Set next word to word with highest prob (argmax)
                    prob, word = torch.max(probs, dim=2) # 1 x bs, 1 x bs

                    #print('prob: ', prob, '\nword_index: ', word_index)
                    #print('word: ', word, '\nword.requires_grad: ', word.requires_grad)

                    # Add word to current sentence
                    word_index = word[0,0].data[0]
                    sent.append(word_index)

                    # End translation if next word is '<s>'
                    if word_index is self.start_token_index or j is 29:
                        sents.append(sent)
                        j = 30 # break out of loop -- stop translating this sentence
            
            return sents
            
            
            
            
            
            

    
# TODO
# - Implement beam search
''')
