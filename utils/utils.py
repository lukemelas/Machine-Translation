import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe
import itertools, os 
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
        with open(os.path.join(self.path, 'model.pkl'), 'w') as f:
            torch.save(f, model_dict)


def AverageMeter():
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

