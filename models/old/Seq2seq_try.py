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

        # Weight tying
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

    def predict(self, src): # legacy implementation
        return self.predict_beam(src)

    def predict_beam(self, src, beam_size=1, TRG='FOR DEBUG'):

        # Final sentence predictions for full batch 
        sents = [] #list (len bs) of list of sentences (word indices)  

        # Encode
        out_e, final_e = self.encoder(src)

        # Loop through batches: make bs = 1 
        for i in range(out_e.size(1)):
            
            # Initial hidden state is last state of encoder
            state_d = tuple(x.select(1,i).unsqueeze(1).contiguous() for x in final_e)
            
            # Extract encoder for attention and source (for padding mask)
            out_e_i = out_e.select(1,i).unsqueeze(1) # for attention
            src_i = src.select(1,i).unsqueeze(1)     

            # Create initial starting word '<s>'
            word_index = self.start_token_index # '<s>'

            # Store tuples of (sent, state, score) in beam
            beam = [([word_index], state_d, 0)]
            
            # Store candidate sentences: length beam_size * vocab_size (really beam_size)
            candidates = [] 

            # Loop until max sentence length reached
            for j in range(50): # max len = 15

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
                    context = self.attention(src_i, out_e_i, out_d)
                    out_cat = torch.cat((out_d, context), dim=2) 

                    # Pass through linear layer
                    x = self.linear1(out_cat)
                    x = self.linear2(x)
                    x = x[0,0] # reshape (1 x 1 x vs --> vs) and do (manual) softmax
                    x = x.exp() / x.exp().sum()
                    
                    # Get top (beam_size) words from distribution
                    probs, words = x.topk(beam_size)
                    for k in range(beam_size):
                        word_index = words[k].data[0]
                        word_score = probs[k].log().data[0] # score = log likelihood
                        
                        # Get sentence probabilities
                        candidate_score = self.sentence_prob(score, word_score, len(sent)) # remove <s>
                        #print('words ', words, 'word_index ', word_index,'probs ', probs, 'candidate_score ', score)
                        candidates.append( (sent + [word_index], state, candidate_score) )

                # Get put top (beam_size) candidates into beam
                beam = sorted(candidates, key=lambda x: x[2])[-(beam_size):] 
                candidates = [] # reset candidates for next beam

                # PRINT PRINT SO SAR
                #print([(' '.join(TRG.vocab.itos[x] for x in c[0]), '{:.3f}'.format(c[2])) for c in beam]) 

            # Put top (1) sentence into final list of sentences
            final_sent = max(beam, key=lambda x: x[2])[0] # extract top sentence
            sents.append(final_sent[1:]) # remove <s> token

        # Return final list of all sentences
        return sents                

    def sentence_prob(self, score, word_score, sent_length, hyperparam_alpha=0.6):
        return score + word_score
        # return (score * ((sent_length - 1) ** hyperparam_alpha) + word_score) / (sent_length ** hyperparam_alpha)

# TODO
# - fix beam search
# - implement badhanau attention

