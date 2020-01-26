from transformers import *
import torch
import torchvision
import random
import torch.nn as nn
import torch.optim as optim
from reader import TrainingInstance, BertPretrainReader
import tokenization
from nerreader import DataReader
from nermodel import NerModel
import argparse

pretrained_bert_name  = 'bert-base-cased'

random_seed = 12345
rng = random.Random(random_seed)
log_path = 'main_logger'
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_train_file', type=str, default='bc2gm_train.tsv', help='training file for ner')
    parser.add_argument('--ner_lr', type=float, default=0.0015, help='Learning rate for ner lstm')
    args = vars(parser.parse_args())
    return args
class BioMLT():
    def __init__(self):
        self.args = parse_args()
        self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name,output_hidden_states=True)
        #print(self.bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.ner_path = self.args['ner_train_file']
        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]
        self.bert_out_dim = self.bert_model.bert.encoder.layer[11].output.dense.out_features
        print("BERT output dim {}".format(self.bert_out_dim))
        self.args['bert_output_dim'] = self.bert_out_dim
        self.bert_optimizer = AdamW(optimizer_grouped_parameters,
                         lr=2e-5)

    def _get_bert_batch_hidden(self, hiddens , bert2toks, layers=[-2,-3,-4]):
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]),0)
        batch_my_hiddens = []
        for means,bert2tok in zip(meanss,bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i,b2t in enumerate(bert2tok):
                if i>0 and b2t!=bert2tok[i-1]:
                    my_hiddens.append(torch.mean(torch.stack(my_token_hids),0))
                    #my_token_hids = [means[i+1]]  ## we skip the CLS token
                    my_token_hids = [means[i]] ## Now I dont skip the CLS token
                else:
                    #my_token_hids.append(means[i+1])
                    my_token_hids = [means[i]] ## Now I dont skip the CLS token
            my_hiddens.append(torch.mean(torch.stack(my_token_hids),0))
            sent_hiddens = torch.stack(my_hiddens)
            batch_my_hiddens.append(sent_hiddens)
        return torch.stack(batch_my_hiddens)

    ##parallel reading not implemented for training
    def pretrain(self):
        file_list = ["PMC6961255.txt"]
        reader = BertPretrainReader(self.bert_tokenizer)
        dataset = reader.create_training_instances(file_list,self.bert_tokenizer)
        tokens = dataset[1].tokens
        logging.info(tokens)
        input_ids = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0) # Batch size 1
        token_type_ids= torch.tensor(dataset[1].segment_ids).unsqueeze(0)
        print(input_ids.shape)
        print(dataset[1].segment_ids)
        next_label = torch.tensor([ 0 if dataset[1].is_random_next  else  1])
        for i in range(10):
            self.bert_optimizer.zero_grad()
            outputs = self.bert_model(input_ids,token_type_ids= token_type_ids, \
                masked_lm_labels=input_ids, next_sentence_label=next_label)
            loss, prediction_scores, seq_relationship_scores = outputs[:3]

            print("Loss {} ".format(loss))
            loss.backward()
            #self.bert_optimizer.step()
    def train_ner(self):
        self.ner_reader = DataReader(self.ner_path, "NER",tokenizer=self.bert_tokenizer)
        self.args['ner_label_vocab'] = self.ner_reader.label_voc
        self.ner_head = NerModel(self.args)

        for i in range(2):
            print("Starting training")
            self.ner_head.optimizer.zero_grad()
            tokens, bert_batch_after_padding, data = self.ner_reader[0]
            sent_lens, masks, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data
            print(bert_batch_ids.shape)
            outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
            bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
            loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
            print(loss.item())
            loss.backward()
            self.ner_head.optimizer.step()
        self.eval_ner()
    def eval_ner(self):
        tokens, bert_batch_after_padding, data = self.ner_reader[1]
        sent_lens, masks, tok_inds, ner_inds,\
             bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data
        outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
        bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
        loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
        print("Predictions ")
        print(torch.argmax(out_logits,dim=2))
        print("Labels")
        print(ner_inds)
if __name__=="__main__":
    biomlt = BioMLT()
    biomlt.train_ner()
