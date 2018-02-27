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
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
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
        x = self.dropout(self.tanh(x))
        x = self.linear2(x)
        return x

    def predict(self, src, beam_size=1): 
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, max_len=30) # returns top beam_size options (as list of tuples)
        beam_outputs.sort(key = lambda x: x[0].data[0], reverse=True) # sort by score
        top1 = list(beam_outputs[0][1].data) # a list of word indices (as ints)
        return top1

    def predict_k(self, src, k, max_len=100):
        '''Predict top k possibilities for first 3 words.'''
        beam_outputs = self.beam_search(src, k, max_len=max_len) # returns top k options (as list of tuples)
        beam_outputs.sort(key = lambda x: x[0].data[0], reverse=True) # sort by score
        topk = [list(option[1].data) for option in beam_outputs] # list of k lists of word indices (as ints)
        return topk

    def beam_search(self, src, beam_size, max_len):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.'''
        # Encode
        source = src.cuda() if use_gpu else batch.src
        outputs_e, states = self.encoder(source) # batch size = 1
        # Start with '<s>'
        initial_score = Variable(torch.zeros(1)).cuda() if use_gpu else Variable(torch.zeros(1)) 
        initial_sent = Variable(torch.LongTensor([self.start_token_index])).cuda() if use_gpu else Variable(torch.LongTensor([self.start_token_index]))
        best_options = [(initial_score, initial_sent, states)] # beam
        # Beam search
        k = beam_size # store best k options
        for length in range(max_len): # maximum target length
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
                    x = self.dropout(self.tanh(x))
                    x = self.linear2(x)
                    x = x.squeeze()
                    probs = x.exp() / x.exp().sum()
                    lprobs = torch.log(probs)
                    # Add top k candidates to options list for next word
                    for index in torch.topk(lprobs, k)[1]: 
                        options.append((torch.add(lprobs[index], lprob), torch.cat([sentence, index]), new_state))
                else: # keep sentences ending in '</s>' as candidates
                    options.append((lprob, sentence, current_state))
            options.sort(key = lambda x: x[0].data[0], reverse=True) # sort options
            best_options = options[:k] # sorts by first element (lprob)
        best_options.sort(key = lambda x: x[0].data[0], reverse=True)
        return best_options

