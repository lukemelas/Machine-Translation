import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .Encoder import EncoderLSTM
from .Decoder import DecoderLSTM

class Attention(nn.Module):
    def __init__(self, attn_type='dot-product', h_dim=300):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        self.h_dim = h_dim

        # Create parameters depending on attention type
        if self.attn_type != 'dot-product' and self.attn_type != 'bahdanau':
            raise Exception('Incorrect attention type')
        elif self.attn_type is 'bahdanau':
            self.linear = nn.Linear(2 * self.h_dim, self.h_dim)
            self.tanh = nn.Tanh()
        #self.softmax = nn.Softmax()

    def forward(self, out_e, out_d):

        # Move batch first
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd

        # Different types of attention
        attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        #if self.attn_type is 'dot-product':
        #    attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        #elif self.attn_type is 'bahdanau':
        #    #attn = self.linear(torch.cat((out_e, out_d), dim=2)) # --> b x  
        #    raise NotImplementedError()

        # Apply attn to encoder outputs
        attn = attn.exp() / attn.exp().sum(dim=1).unsqueeze(1) # turn this into softmax with updated pytorch
        context = attn.transpose(1,2).bmm(out_e) # --> b x lt x hd
        context = context.transpose(0,1) # --> lt x b x hd
        return context


class Seq2seq(nn.Module):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers, dropout_p, start_token_index=0, eos_token_index=1):
        super(Seq2seq, self).__init__()
        self.h_dim = h_dim
        self.vocab_size_trg = embedding_trg.size(0)
        self.start_token_index = start_token_index
        self.eos_token_index = eos_token_index

        # Create encoder, decoder, attention
        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p)
        self.decoder = DecoderLSTM(embedding_trg, h_dim, num_layers, dropout_p=dropout_p)
        self.attention = Attention()

        # Create linear layers to combine context and hidden state
        self.linear1 = nn.Linear(2 * self.h_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.vocab_size_trg)

        # Weight tying
        if False and self.decoder.embedding.weight.size().equals(linear2.weight.size()): # weight tying
            self.linear2.weight = self.decoder.embedding.weight

    def forward(self, src, trg):
        
        # Pass through encoder
        out_e, final_e = self.encoder(src)

        # Pass through decoder
        out_d, final_d = self.decoder(trg, final_e)

        # Apply attention
        context = self.attention(out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2) 

        # Pass through linear layers to predict next word and return word probabilities
        x = self.linear1(out_cat)
        x = self.linear2(x)
        return x

    def predict(self, src, trg=None):

        return self.predict_beam(src)

        # Store predictions: list of len bs of items of size sl x vs
        sents = []

        # Pass src through encoder
        out_e, final_e = self.encoder(src)

        # Loop through batches: for the inner code, bs = 1
        for i in range(out_e.size(1)):
            
            # Initial hidden state is last state of encoder
            state_d = tuple(x.select(1,i).unsqueeze(1).contiguous() for x in final_e)
            out_e_i = out_e.select(1,i).unsqueeze(1) # for attention
                
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

                # Pass through decoder one word at a time
                out_d, state_d = self.decoder(word, state_d)

                # Apply attention
                context = self.attention(out_e_i, out_d)
                out_cat = torch.cat((out_d, context), dim=2) 

                # Pass through linear layers to predict next word and return word probabilities
                x = self.linear1(out_cat)
                x = self.linear2(x)
                
                # Sample word from distribution (by taking maximum)
                probs, word = x.topk(1)
                word = word.view(1,1)
                word_index = word[0,0].data[0]

                # Add word to current sentence
                sent.append(word_index)
                
                # Update word counter
                j += 1

            # Append generated sentence to full list of sentences
            sents.append(sent)

        # Return list of sentences (list of list of words)
        return sents

    def predict_beam(self, src, beam_size=5):

        # Store final sentences: list of len bs of lists (of variable len) of words
        sents = []

        # Pass src through encoder
        out_e, final_e = self.encoder(src)

        # Loop through batches: for the inner code, bs = 1
        for i in range(out_e.size(1)):
            
            # Initial hidden state is last state of encoder
            state_d = tuple(x.select(1,i).unsqueeze(1).contiguous() for x in final_e)
            out_e_i = out_e.select(1,i).unsqueeze(1) # for attention
                
            # Create initial starting word '<s>'
            word_index = self.start_token_index # '<s>'

            # Store tuples of (sent, state, score) in beam
            beam = [([word_index], state_d, 1)]
            
            # Store candidate sentences: length beam_size * vocab_size (really beam_size)
            candidates = [] 

            # Loop until max sentence length reached
            for j in range(20): # max len = 15

                # Stop if all sentences in beam end in '</s>'
                stop = True
                for sent, state, score in beam: # stop if all sentences 
                    stop = stop and (sent[-1] is self.eos_token_index)
                if stop is True: break
                
                # Loop through sentences in beam
                for sent, state, score in beam:
                    word = sent[-1]

                    # If sentence already finished, keep it as a candidate
                    if word is self.eos_token_index:
                        candidates.append( (sent, state, score) )
                        continue

                    # Otherwise run through decoder
                    word = Variable(torch.LongTensor([word_index]).view(1,1), requires_grad=False)
                    if use_gpu: word = word.cuda()
                    out_d, state = self.decoder(word, state)

                    # Apply attention 
                    context = self.attention(out_e_i, out_d)
                    out_cat = torch.cat((out_d, context), dim=2) 

                    # Pass through linear layer
                    x = self.linear1(out_cat)
                    x = self.linear2(x)
                    
                    # Sample word from distribution (by taking maximum)
                    probs, words = x.topk(beam_size)
                    for k in range(beam_size):
                        word_index = words[k].data[0]
                        word_score = probs[k].data[0]
                        
                        # Get sentence probabilities
                        candidate_score = self.sentence_prob(score, word_score, len(sent))
                        print('words ', words, 'word_index ', word_index, 'state ', state, 'candidate_score ', score)
                        candidates.append( (sent + word_index, state, candidate_score) )

                # Get put top (beam_size) candidates into beam
                beam = sorted(candidates, key=lambda x: x[2])[-(beam_size):] 
                candidates = [] # reset candidates for next beam

            # Put top (1) sentence into final list of sentences
            sents.append(sent)

        # Return final list of all sentences
        return sents                

    def sentence_prob(self, score, word_score, sent_length):
        return score * word_score

# TODO
# - implement beam search
# - implement attention
# - implement other type of attention
# - implement dropout

