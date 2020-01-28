from transformers import *
import logging
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import os
log_path = "trial_log"
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokens = tokenizer.tokenize("[MASK] [CLS] [SEP] [UNK]  [PAD]")

special_tokens = ['[MASK]', '[CLS]', '[SEP]', '[UNK]', '[PAD]']
special_token_ids = [103, 101, 102, 100, 0]


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer=tokenizer,file_path="PMC6961255.txt", block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0]

        self.examples = tokenizer.batch_encode_plus(lines, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

linebyline_dataset = LineByLineTextDataset()

for i in range(10):
    print("Line by line {} ".format(linebyline_dataset[i]))
