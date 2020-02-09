from transformers import *
#import tensorflow as tf
import logging
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import os
import argparse
log_path = "trial_log"
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokens = tokenizer.tokenize("[MASK] [CLS] [SEP] [UNK]  coronavirus coronavirus coronavirus [PAD]")

special_tokens = ['[MASK]', '[CLS]', '[SEP]', '[UNK]', '[PAD]']
special_token_ids = [103, 101, 102, 100, 0]


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--train_data_file", default=None, type=str, required=False, help="The input training data file (a text file)."
)

parser.add_argument("--load_model",action="store_true",default=False)

args = parser.parse_args()

if not args.load_model:
     print("Arda is my girl")

biobert_model = BertModel.from_pretrained("../biobert_data/biobert_v1.1_pubmed",from_tf=True,output_hidden_states=True)
bert_cased_model = BertModel.from_pretrained("bert-base-cased",output_hidden_states=True)

toks = tokenizer.encode(tokens)
inp = torch.tensor(tokenizer.encode(tokens)).unsqueeze_(0)
print(toks)
biobert_out = biobert_model(inp)
bert_case_out = bert_cased_model(inp)
print("Biobert out {}".format(biobert_out[-1][-1][0,-3,:10]))
print("Bert base cased out {}".format(bert_case_out[-1][-1][0,-3,:10]))


