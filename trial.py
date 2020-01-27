from transformers import *
import logging
import torch
log_path = "trial_log"
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokens = tokenizer.tokenize("[MASK] [CLS] [SEP] [UNK]  [PAD]")

special_tokens = ['[MASK]', '[CLS]', '[SEP]', '[UNK]', '[PAD]']
special_token_ids = [103, 101, 102, 100, 0]
inds = torch.randn(118)
mask = torch.ones(inds.shape)
mask[[0,1,2]] = torch.tensor([10.0,200,300])
print(mask)
