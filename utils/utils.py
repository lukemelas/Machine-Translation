import itertools, os, re
import tempfile, subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe
use_gpu = torch.cuda.is_available()

class Logger():
    '''Prints to a log file and to standard output''' 
    def __init__(self, path):
        if os.path.exists(path):
            self.path = path
        else:
            raise Exception('path does not exist')
    
    def log(self, info, stdout=True):
      with open(os.path.join(self.path, 'log.log'), 'a') as f:
        print(info, file=f)
      if stdout:
        print(info)
    
    def save_model(self, model_dict):
        #with open(os.path.join(self.path, 'model.pkl'), 'w') as f:
        torch.save(model_dict, os.path.join(self.path, 'model.pkl'))

class AverageMeter():
    '''Computes and stores the average and current value. 
       Taken from the PyTorch ImageNet tutorial'''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
   
    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

def moses_multi_bleu(outputs, references, lw=False):
    '''Outputs, references are lists of strings. Calculates BLEU score using https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl -- Python function from Google '''
    
    # Save outputs and references as temporary text files
    out_file = tempfile.NamedTemporaryFile()
    out_file.write('\n'.join(outputs).encode('utf-8'))
    out_file.write(b'\n')
    out_file.flush() #?
    ref_file = tempfile.NamedTemporaryFile()
    ref_file.write('\n'.join(references).encode('utf-8'))
    ref_file.write(b'\n')
    ref_file.flush() #?
    
    # Use moses multi-bleu script
    with open(out_file.name, 'r') as read_pred:
        bleu_cmd = ['scripts/multi-bleu.perl']
        bleu_cmd = bleu_cmd + ['-lc'] if lw else bleu_cmd
        bleu_cmd = bleu_cmd + [ref_file.name]
        try: 
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode('utf-8')
            #print(bleu_out)
            bleu_score = float(re.search(r'BLEU = (.+?),', bleu_out).group(1))
        except subprocess.CalledProcessError as error:
            print(error)
            raise Exception('Something wrong with bleu script')
            bleu_score = 0.0
    
    # Close temporary files
    out_file.close()
    ref_file.close()
   
    return bleu_score

