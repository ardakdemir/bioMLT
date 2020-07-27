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
        if not hasattr(self.args, "ner_label_dim"):
            self.label_dim = len(args.ner_label_vocab)
        else:
            self.label_dim = self.args.ner_label_dim
        self.ner_drop = 0.3 if not hasattr(self.args, "ner_drop") else self.args.ner_drop
        self.num_labels = self.label_dim
        self.output_dim = self.label_dim * self.label_dim
        self.device = args.device
        if self.args.crf:
            logging.info("Using NER with CRF")
            self.classifier = CRF(self.input_dims, self.num_labels, self.device)
            self.loss = CRFLoss(self.num_labels, device=self.device)
        else:
            self.classifier = nn.Linear(self.input_dims, self.output_dim)
            self.loss = CrossEntropyLoss(ignore_index=PAD_IND)
        self.lr = args.ner_lr
        self.optimizer = optim.AdamW([{"params": self.classifier.parameters()}], \
                                     lr=self.lr, eps=1e-6)

        self.dropout = nn.Dropout(self.ner_drop)

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
    def forward(self, batch, labels=None, pred=False, loss_aver=True):
        out_logits = self.classifier(batch)
        batch_size = batch.shape[0]
        if self.dropout and not pred:
            out_logits = self.dropout(out_logits)
        # print(out_logits.shape)
        # print(labels.shape)a
        if pred:
            if labels is not None:
                loss = -1
                if self.args.crf:
                    lengths = torch.sum((labels > self.num_labels), axis=1)
                    loss = self.loss(out_logits, labels, lengths)
                else:
                    loss = self.loss(out_logits.view(-1, self.output_dim), labels.view(-1))
                if loss_aver:
                    loss = loss / batch_size
                    print(" Loss {} batch size {} ".format(loss.item(), batch_size))
                return out_logits, loss.item()
            return out_logits
        if labels is not None:
            ## view tehlikeli bir hareket!!!!
            if self.args.crf:
                lengths = torch.sum((labels > self.num_labels), axis=1)
                loss = self.loss(out_logits, labels, lengths)
            else:
                loss = self.loss(out_logits.view(-1, self.output_dim), labels.view(-1))
            if loss_aver:
                loss = loss / batch_size
                print(" Loss {} batch size {} ".format(loss.item(),batch_size))
            return loss, out_logits

        return out_logits
