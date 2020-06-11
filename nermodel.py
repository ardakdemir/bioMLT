import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import os
import copy
import logging
from transformers import *
class NerModel(nn.Module):
    def __init__(self,args):
        super(NerModel,self).__init__()
        self.args = args
        self.input_dims = args.bert_output_dim
        self.label_voc = args.ner_label_vocab
        self.output_dim = len(self.label_voc)

        # Now I am calculating one-dimensional labels so taking the square of the label vocab
        # Treats transitions between tags as a different tag
        # Ignore transitions because the labels are different
        # e.g LOC -> ORG   !=  O -> ORG
        self.classifier = nn.Linear(self.input_dims,self.output_dim )
        self.loss = CrossEntropyLoss()
        self.lr = args.ner_lr
        self.optimizer = optim.AdamW([{"params": self.classifier.parameters()}],\
        lr=self.lr, eps=1e-6)

    # add the attention masks to exclude cls and pad etc.
    def forward(self,batch,labels = None,pred=False):
        out_logits = self.classifier(batch)
        #print(out_logits.shape)
        #print(labels.shape)a
        if pred:
            return out_logits
        if labels is not None:
            ## view tehlikeli bir hareket!!!!
            loss = self.loss(out_logits.view(-1,self.output_dim),labels.view(-1))
            return loss,out_logits
        return out_logits
