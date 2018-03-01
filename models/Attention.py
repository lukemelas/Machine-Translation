import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class Attention(nn.Module):
    def __init__(self, pad_token=1, bidirectional=True, attn_type='dot-product', h_dim=300):
        super(Attention, self).__init__()
        self.bidirectional = bidirectional
        self.attn_type = attn_type
        self.h_dim = h_dim
        self.pad_token = pad_token

        # Create parameters depending on attention type
        if self.attn_type != 'dot-product' and self.attn_type != 'additive':
            raise Exception('Incorrect attention type')
        elif self.attn_type is 'additive':
            self.linear = nn.Linear(2 * self.h_dim, self.h_dim)
            self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, in_e, out_e, out_d):

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional: # sum hidden states for both directions
            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd

        # Different types of attention
        if self.attn_type == 'dot-product':
            attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        elif self.attn_type == 'additive':
            # Resize output tensors for efficient matrix multiplication, then apply additive attention
            bs_sl_tl_hdim = (out_e.size(0), out_e.size(1), out_d.size(1), out_e.size(2))
            out_e = out_e.unsqueeze(2).expand(bs_sl_tl_hdim) # b x sl x tl x hd
            out_d = out_d.unsqueeze(1).expand(bs_sl_tl_hdim) # b x sl x tl x hd 
            attn = self.linear(torch.cat((out_e, out_d), dim=2)) # --> b x sl x tl

        # Apply attn to encoder outputs
        # attn = torch.nn.functional.softmax(attn)
        attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True) # in updated pytorch, make softmax
        attn = attn.transpose(1,2) # --> b x tl x sl
        context = attn.bmm(out_e) # --> b x lt x hd
        context = context.transpose(0,1) # --> lt x b x hd
        return context

