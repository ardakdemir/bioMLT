import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torch
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
class QasModel(nn.Module):
    def __init__(self,args):
        super(QasModel,self).__init__()
        self.args = args
        self.input_dims = self.args.bert_output_dim
        #self.label_voc = args['ner_label_vocab']
        self.output_dim = self.args.qas_out_dim
        ## now I am calculating one-dimensional labels so taking the square of the label vocab
        self.qa_outputs = nn.Linear(self.input_dims,self.output_dim)
        self.loss = CrossEntropyLoss()
        self.lr = self.args.qas_lr
        self.optimizer = optim.AdamW([{"params": self.qa_outputs.parameters()}],\
        lr=self.lr, eps=self.args.qas_adam_epsilon)
    ## add the attention masks to exclude CLS and PAD etc.
    def forward(
        self,bert_outputs,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        ## now getting the final hidden layer output!!
        ## we might want to average over 4-5 layers?!?!
        sequence_output = bert_outputs

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs

    def predict(self,bert_outputs,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,train=False):
        if not train:
            with torch.no_grad():
                logits = self.qa_outputs(bert_outputs)
        else:
            logits = self.qa_outputs(bert_outputs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_preds = torch.argmax(start_logits,dim=-1)
        end_preds = torch.argmax(end_logits,dim=-1)
        return (start_preds, end_preds,)
