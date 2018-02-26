import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from .Encoder import EncoderLSTM
from .Decoder import DecoderLSTM
from .Attention import Attention

class Seq2seq(nn.Module):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers, dropout_p, bi, attn_type, start_token_index=0, eos_token_index=2, pad_token_index=1):
        super(Seq2seq, self).__init__()
        # Store hyperparameters
        self.h_dim = h_dim
        self.vocab_size_trg = embedding_trg.size(0)
        self.start_token_index = start_token_index
        self.eos_token_index = eos_token_index
        # Create encoder, decoder, attention
        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p, bidirectional=bi)
        self.decoder = DecoderLSTM(embedding_trg, h_dim, num_layers * 2 if bi else num_layers, dropout_p=dropout_p)
        self.attention = Attention(pad_token=pad_token_index, bidirectional=bi, attn_type=attn_type)
        # Create linear layers to combine context and hidden state
        self.linear1 = nn.Linear(2 * self.h_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.vocab_size_trg)
        # Tie weights of decoder embedding and output 
        if True and self.decoder.embedding.weight.size() == self.linear2.weight.size(): # weight tying
            print('Weight tying!')
            self.linear2.weight = self.decoder.embedding.weight

    def forward(self, src, trg):
        # Encode
        out_e, final_e = self.encoder(src)
        # Decode
        out_d, final_d = self.decoder(trg, final_e)
        # Attend
        context = self.attention(src, out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2) 
        # Predict (returns probabilities)
        x = self.linear1(out_cat)
        x = self.linear2(x)
        return x

    def predict(self, src): 
        '''Predict using greedy search'''
        return self.predict_beam(src, beam_size=1)

    def predict_beam(self, src, beam_size=1):
        '''Predict using beam search. Works only when src has batch size 1: src.size(1)=1'''
        # Encode
        source = src.cuda() if use_gpu else batch.src
        outputs_e, states = self.encoder(source) # batch size = 1
        # Start with '<s>'
        initial_sent = Variable(torch.zeros(1)).cuda() if use_gpu else Variable(torch.zeros(1)) 
        initial_start = Variable(torch.LongTensor([self.start_token_index])).cuda() if use_gpu else Variable(torch.LongTensor([self.start_token_index]))
        best_options = [(initial_sent, initial_start, states)] # beam
        # Beam search
        k = beam_size # store best k options
        for length in range(50): # maximum target length
            options = [] # candidates 
            for lprob, sentence, current_state in best_options:
                last_word = sentence[-1]
                if last_word.data[0] != self.eos_token_index:
                    # Decode
                    outputs_d, new_state = self.decoder(last_word.unsqueeze(1), current_state)
                    # Attend
                    context = self.attention(source, outputs_e, outputs_d)
                    out_cat = torch.cat((outputs_d, context), dim=2)
                    x = self.linear1(out_cat)
                    x = self.linear2(x)
                    x = x.squeeze()
                    probs = x.exp() / x.exp().sum()
                    # Add top k candidates to options list for next word
                    for index in torch.topk(probs, k)[1]: 
                        options.append((torch.add(probs[index], lprob), torch.cat([sentence, index]), new_state))
                else: # keep sentences ending in '</s>' as candidates
                    options.append((lprob, sentence, current_state))
            options.sort(key = lambda x: x[0].data[0], reverse=True) # sort options
            best_options = options[:k] # sorts by first element (lprob)
        best_options.sort(key = lambda x: x[0].data[0], reverse=True)
        best_choice = best_options[0] # best overall sentence
        sentence = best_choice[1].data
        out = list(sentence) # return list of word indices (as ints)
        return out

# TODO

