import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import logging

from nerreader import PAD, UNK, START_TAG, END_TAG, PAD_IND, END_IND, START_IND, UNK_IND

from crf import CRF, CRFLoss
from transformers import *


class NerModel(nn.Module):
    def __init__(self, args):
        super(NerModel, self).__init__()
        self.args = args
        self.input_dims = args.bert_output_dim
        self.label_voc = args.ner_label_vocab
        self.num_labels = len(self.label_voc)
        self.output_dim = len(self.label_voc) * len(self.label_voc)
        self.device = args.device
        # Now I am calculating one-dimensional labels so taking the square of the label vocab
        # Treats transitions between tags as a different tag
        # Ignore transitions because the labels are different
        # e.g LOC -> ORG   !=  O -> ORG
        # self.classifier = nn.Linear(self.input_dims, self.output_dim)
        # self.loss = CrossEntropyLoss()
        if self.args.crf:
            logging.info("Using CRF with NER")
            logging.info("Using NER with CRF")
            self.classifier = CRF(self.input_dims, self.num_labels, self.device)
            self.loss = CRFLoss(self.num_labels, device = self.device)
        else:
            self.classifier = nn.Linear(self.input_dims, self.output_dim)
            self.loss = CrossEntropyLoss(ignore_index = PAD_IND)
        self.lr = args.ner_lr
        self.optimizer = optim.AdamW([{"params": self.classifier.parameters()}], \
                                     lr=self.lr, eps=1e-6)

    def _viterbi_decode(self, feats, sent_len):
        start_ind = START_IND
        end_ind = END_IND
        # feats = feats[:,end_ind+1:,end_ind+1:]
        parents = [[torch.tensor(start_ind) for x in range(feats.size()[1])]]
        layer_scores = feats[1, :, start_ind]

        for feat in feats[2:sent_len, :, :]:
            # layer_scores =feat[:,:start_ind,:start_ind] + layer_scores.unsqueeze(1).expand(1,layer_scores.shape[1],layer_scores.shape[2])
            layer_scores = feat + layer_scores.unsqueeze(0).expand(layer_scores.shape[0], layer_scores.shape[0])
            layer_scores, parent = torch.max(layer_scores, dim=1)
            parents.append(parent)
        # layer_scores = layer_scores + self.crf.transitions[self.l2ind[END_TAG],:]

        path = [end_ind]
        path_score = layer_scores[end_ind]
        parent = path[0]
        # parents.reverse()
        for p in range(len(parents) - 1, -1, -1):
            path.append(parents[p][parent].item())
            parent = parents[p][parent]
        path.reverse()
        return path, path_score.item()

    # add the attention masks to exclude cls and pad etc.
    def forward(self, batch, labels=None, pred=False):
        out_logits = self.classifier(batch)
        # print(out_logits.shape)
        # print(labels.shape)a
        if pred:
            return out_logits
        if labels is not None:

            ## view tehlikeli bir hareket!!!!
            if self.args.crf:
                lengths = torch.sum((labels > self.num_labels), axis=1)
                loss = self.loss(out_logits, labels, lengths)
            else:
                loss = self.loss(out_logits.view(-1, self.output_dim), labels.view(-1))
            return loss, out_logits

        return out_logits
