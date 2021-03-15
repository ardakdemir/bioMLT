from collections import Counter
import glob
import torch
import numpy as np
import logging
from transformers import BertTokenizer
# from parser.parsereader import bert2token, pad_trunc_batch
from vocab import Vocab, VOCAB_PREF
from utils import sort_dataset, unsort_dataset

PAD = "[PAD]"
START_TAG = "[CLS]"
END_TAG = "[SEP]"
UNK = "[UNK]"

PAD_IND = 0
START_IND = 1
END_IND = 2
UNK_IND = 3


def all_num(token):
    n = "0123456789."
    for c in token:
        if c not in n:
            return False
    return True


def bert2token(my_tokens, bert_tokens, bert_ind=0):
    """
        Given input tokens combines bert_tokens to match the input tokens
    """
    inds = []
    token_sum = ""
    bert_ind = bert_ind
    for ind in range(len(my_tokens)):
        my_token = my_tokens[ind]
        token = bert_tokens[bert_ind]
        if token == UNK:  # UNK token handler
            inds.append(ind)
            bert_ind = bert_ind + 1
            continue
        while len(token_sum) != len(my_token) and bert_ind < len(bert_tokens):
            token = bert_tokens[bert_ind]
            if token.startswith("##"):
                token_sum += token[2:]
            else:
                token_sum += token
            inds.append(ind)
            bert_ind += 1
        if not len(token_sum) == len(my_token):
            print("PROBLEMATIC TOKENIZATION")
            print("{} {} {} {}".format(my_token, token_sum, my_tokens, bert_tokens))
            return
        token_sum = ""

    return inds, ind + 1


def pad_trunc_nerdata_batch(batch, max_len, pad=PAD, pad_ind=PAD_IND):
    padded_batch = []
    sent_lens = []
    for sent in batch:
        sent_lens.append(len(sent))
        if len(sent) >= max_len:
            padded_batch.append(sent)
        else:
            l = len(sent)
            index_len = len(sent[0])
            padded_sent = sent
            for i in range(max_len - l):
                padded_sent.append([PAD for x in range(index_len)])  ## PAD ALL FIELDS WITH [PAD]
            padded_batch.append(padded_sent)
    return padded_batch, sent_lens


def pad_trunc_batch(batch, max_len, pad=PAD, pad_ind=PAD_IND, bert=False, b2t=False):
    padded_batch = []
    sent_lens = []
    for sent in batch:
        sent_lens.append(len(sent))
        if len(sent) >= max_len:
            if bert:
                if b2t:
                    padded_batch.append(sent)
                else:
                    # padded_batch.append(sent + ["[SEP]"])
                    padded_batch.append(sent)
            else:
                padded_batch.append(sent)
        else:
            l = len(sent)
            if not bert:
                index_len = len(sent[0])
            padded_sent = sent
            for i in range(max_len - l):
                if bert:
                    if b2t:
                        padded_sent.append(padded_sent[-1])
                    else:
                        padded_sent = padded_sent + [PAD]
                else:
                    padded_sent.append([PAD for x in range(index_len)])  ## PAD ALL FIELDS WITH [PAD]
            if bert:
                if not b2t:
                    # padded_sent = padded_sent + ["[SEP]"]
                    p = 10
            padded_batch.append(padded_sent)
    return padded_batch, sent_lens


def group_into_batch(dataset, batch_size):
    """

        Batch size is given in word length so that some batches do not contain
        too many examples!!!

        Do not naively batch by number of sentences!!

    """

    batched_dataset = []
    sentence_lens = []
    current_len = 0
    i = 0

    ## they are already in sorted order
    current = []
    max_len = 0
    print("Batch size: {}".format(batch_size))
    for x in dataset:
        current.append(x)
        max_len = max(len(x), max_len)  ##
        current_len += len(x)
        if len(x) > 200:
            logging.info("Length {}".format(len(x)))
            logging.info(x)
        if current_len > batch_size:
            # print(current)
            current, lens = pad_trunc_nerdata_batch(current, max_len)
            batched_dataset.append(current)
            sentence_lens.append(lens)
            current = []
            current_len = 0
            max_len = 0
    if len(current) > 0:
        current, lens = pad_trunc_batch(current, max_len)
        sentence_lens.append(lens)
        batched_dataset.append(current)
    return batched_dataset, sentence_lens


def get_orthographic_feat(token):
    if token == START_TAG or token == END_TAG or token == PAD:
        return 5
    if "'" in token:
        return 4
    if all_num(token):
        return 3
    if token.isupper():
        return 2
    if token.istitle():
        return 1
    if token.islower():
        return 0
    return 0


def pad_trunc(sent, max_len, pad_len, pad_ind):
    if len(sent) > max_len:
        return sent[:max_len]
    else:
        l = len(sent)
        for i in range(max_len - l):
            if pad_len == 1:
                sent.append(0)
            else:
                sent.append([0 for i in range(pad_len)])
        return sent


def ner_document_reader(file_path, sent_len=None):
    document = ""
    print("Readdinng {}".format(file_path) )
    with open(file_path, "r") as f:
        f = f.read()
        doc = f.split("\n\n")
        if sent_len is not None:
            doc = doc[:sent_len]
        sents = [" ".join([token.split()[0] for token in sent.split("\n")[:-1]]) for sent in doc if len(sent) > 1]
        for sent in sents:
            document += sent + "\n"
    return document


class DataReader:

    def __init__(self, file_path, task_name, tokenizer, batch_size=300, for_eval=False, crf=False):
        self.for_eval = for_eval
        self.file_path = file_path
        self.crf = crf  # Generate 2-d labels
        if self.crf:
            print("Generating 2-d labels")
        self.task_name = task_name
        self.batch_size = batch_size
        self.dataset, self.orig_idx, self.label_counts = self.get_dataset()
        print("Dataset size : {}".format(len(self.dataset)))
        print("NER Label Counts: {}".format(self.label_counts))
        self.data_len = len(self.dataset)
        self.l2ind, self.word2ind, self.vocab_size = self.get_vocabs()
        # self.pos_voc = Vocab(self.pos2ind)
        self.label_vocab = Vocab(self.l2ind)
        self.word_vocab = Vocab(self.word2ind)
        self.batched_dataset, self.sentence_lens = group_into_batch(self.dataset, batch_size=self.batch_size)
        self.for_eval = for_eval
        self.num_cats = len(self.l2ind)
        print("Number of NER categories: {}".format(self.num_cats))
        self.bert_tokenizer = tokenizer
        self.val_index = 0

    def get_ind2sent(self, sent):
        return " ".join([self.word2ind[w] for w in sent])

    def get_bert_input(self, batch_size=1, morp=False, for_eval=False):
        if for_eval:
            indexes = [i % self.data_len for i in range(self.val_index, self.val_index + batch_size)]
            self.val_index += batch_size
            self.val_index %= self.data_len
        else:
            indexes = np.random.permutation([i for i in range(self.data_len)])
            indexes = indexes[:batch_size]
        sents, labels = self.get_sents(indexes, feats=morp)
        bert_inputs = []
        for sent, label in zip(sents, labels):
            my_tokens = [x[0] for x in sent]
            sentence = " ".join(my_tokens)
            marked_sent = "[CLS]" + sentence + "[SEP]"
            bert_tokens = self.bert_tokenizer.tokenize(marked_sent)
            ids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
            enc_ids = self.bert_tokenizer.encode(sentence)
            seq_ids = [1 for i in range(len(bert_tokens))]
            bert2tok, final_len = bert2token(my_tokens, bert_tokens)
            lab = self.prepare_label(label, self.l2ind)
            bert_inputs.append([torch.tensor([ids], dtype=torch.long), torch.tensor(enc_ids, dtype=torch.long), \
                                torch.tensor([seq_ids], dtype=torch.long), torch.tensor(bert2tok), lab])
        return my_tokens, bert_tokens, bert_inputs

    def get_vocabs(self):
        l2ind = {PAD: PAD_IND, START_TAG: START_IND, END_TAG: END_IND}
        word2ix = {PAD: PAD_IND, START_TAG: START_IND, END_TAG: END_IND}
        # pos2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND}
        # l2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND, ROOT_TAG:PAD_IND }
        # word2ix = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND ,ROOT_TAG:PAD_IND}
        # pos2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND ,ROOT_TAG:PAD_IND}
        # print(self.label_counts)

        for x in self.label_counts:
            if x not in l2ind:
                l2ind[x] = len(l2ind)

        for sent in self.dataset:
            for word in sent:
                try:
                    a = word[0]
                except :
                    print("Problem during vocab reading")
                    print("Sentence: {}".format(sent))
                    print("Word: {}".format(word))
                if word[0] not in word2ix:
                    word2ix[word[0]] = len(word2ix)
        vocab_size = len(word2ix)
        return l2ind, word2ix, vocab_size

    def get_dataset(self):
        dataset = open(self.file_path, encoding='utf-8').readlines()
        new_dataset = []
        first_line = dataset[0].split()
        ind = 1
        while len(first_line) == 0:
            first_line = dataset[ind].split()
            ind = ind + 1

        sent = []
        cropped_long_sentence = 0
        label_counts = Counter()
        root = [START_TAG for x in range(len(first_line))]
        end_tags = [END_TAG for x in range(len(first_line))]
        for line in dataset:
            if len(line.strip()) == 0:
                if len(sent) > 0:
                    sent.append(end_tags)
                    if len(sent) > 2 and len(sent) < 200:
                        new_dataset.append([root] + sent)
                    elif len(sent) > 200:
                        cropped_long_sentence += 1
                    # new_dataset.append(sent)
                    sent = []
            else:
                if len(line.strip()) < 2:
                    continue
                else:
                    row = line.rstrip().split()
                    row[0] = row[0].replace("\ufeff", "")
                    sent.append(row)
                    label_counts.update([row[-1]])
        if len(sent) > 2 and len(sent) < 200:
            sent.append(end_tags)
            new_dataset.append([root] + sent)
            # new_dataset.append(sent)
        elif len(sent) > 200:
            cropped_long_sentence += 1
        print("Number of sentences : {} ".format(len(new_dataset)))
        print("Cropped long sentences for {}  : {} ".format(self.file_path, cropped_long_sentence))
        # print(new_dataset)
        new_dataset, orig_idx = sort_dataset(new_dataset, sort=True)
        print("Label counts {}".format(label_counts))
        return new_dataset, orig_idx, label_counts

    def get_next_data(sent_inds, data_len=-1, feats=True, padding=False):
        sents, labels = self.get_sents(sent_inds, feats=feats)
        datas = []
        labs = []
        for sent, label in zip(sents, labels):
            sent_vector = []
            lab_ = []
            for l, word in zip(label, sent):
                if data_len != -1 and data_len == len(lab_):
                    break
                sent_vector.append(self.getword2vec(word))
                lab_.append(l2ind[l])
            if padding:
                sent_vector = pad_trunc(sent_vector, max_len=100, pad_len=400)
                lab_ = pad_trunc(lab_, max_len=100, pad_len=1)
            datas.append(sent_vector)
            labs.append(lab_)
        return torch.tensor(np.array(datas)).float(), torch.tensor(labs[0], dtype=torch.long).view(-1)

    def get_sents(self, sent_inds, feats=False, label_index=-1):
        sents = []
        labels = []
        for i in sent_inds:
            sent = []
            label = []
            for y in self.dataset[i]:
                if feats:
                    sent.append([y[0], y[1]])
                else:
                    sent.append([y[0]])
                label.append(y[label_index])
            sents.append(sent)
            labels.append(label)
        return sents, labels

    ## compatible with getSent and for word embeddings
    def prepare_sent(self, sent, word2ix):
        idx = [word2ix[word[0]] for word in sent]
        return torch.tensor(idx, dtype=torch.long)

    def prepare_label(self, labs, l2ix):
        idx = [l2ix[lab] for lab in labs]
        return torch.tensor(idx, dtype=torch.long)

    def getword2vec(self, row):
        key = row[0].lower()
        root = row[1][:row[1].find("+")].encode().decode("unicode-escape")
        while (len(key) > 0):
            if key in word_vectors:
                return word_vectors[key]
            elif root.lower() in word_vectors:
                return word_vectors[root.lower()]
            else:
                return word_vectors["OOV"]
        return 0

    def get_1d_targets(self, targets):
        prev_tag = self.l2ind[START_TAG]
        tagset_size = self.num_cats
        targets_1d = []
        for current_tag in targets:
            targets_1d.append(current_tag * (tagset_size) + prev_tag)
            prev_tag = current_tag
        return targets_1d

    def getword2vec2(self, row):
        key = row[0].lower()
        root = row[1][:row[1].find("+")].encode().decode("unicode-escape")  ## for turkish special chars
        while (len(key) > 0):
            if key in word_vectors:
                return 2
            elif root.lower() in word_vectors:
                return 1
            else:
                return 0
        return 0

    def __len__(self):
        return len(self.batched_dataset)

    def __getitem__(self, idx, random=True):
        """
            Indexing for the DepDataset
            converts all the input into tensor before passing

            input is of form :
                word_ids  (Batch_size, Sentence_lengths)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not self.for_eval:
            if random:
                idx = np.random.randint(len(self.batched_dataset))
            idx = idx % len(self.batched_dataset)
        batch = self.batched_dataset[idx]
        lens = self.sentence_lens[idx]
        tok_inds = []
        ner_inds = []
        tokens = []
        for x in batch:
            data = list(zip(*x)) ##unzip the batch
            toks,labels = data[0],data[-1]
            tokens.append(toks)
            # pos_inds.append(self.pos_vocab.map(poss))
            tok_inds.append(self.word_vocab.map(toks))
            if self.crf:
                ner_inds.append(self.get_1d_targets(self.label_vocab.map(labels)))
            else:
                ner_inds.append(self.label_vocab.map(labels))
        assert len(tok_inds) == len(ner_inds) == len(tokens) == len(batch)
        for toks in tokens:
            if toks[0] != "[CLS]":
                logging.info("Problemli batch")
                logging.info(tokens)
                break
        tok_inds = torch.LongTensor(tok_inds)
        ner_inds = torch.LongTensor(ner_inds)
        # pos_inds = torch.LongTensor(pos_inds)
        bert_batch_before_padding = []
        bert_lens = []
        max_bert_len = 0
        bert2toks = []
        cap_types = []
        masks = torch.ones(tok_inds.shape, dtype=torch.bool)
        i = 0
        for sent, l in zip(batch, lens):
            my_tokens = [x[0] for x in sent]
            cap_types.append(torch.tensor([get_orthographic_feat(x[0]) for x in sent]))
            sentence = " ".join(my_tokens)
            masks[i, :l] = torch.tensor([0] * l, dtype=torch.bool)
            i += 1
            bert_tokens = self.bert_tokenizer.tokenize(sentence)
            bert_lens.append(len(bert_tokens))
            # bert_tokens = ["[CLS]"] + bert_tokens
            max_bert_len = max(max_bert_len, len(bert_tokens))
            ## bert_ind = 0 since we already put CLS as the SOS  token
            b2tok, ind = bert2token(my_tokens, bert_tokens, bert_ind=0)
            assert ind == len(my_tokens), "Bert ids do not match token size"
            bert_batch_before_padding.append(bert_tokens)
            bert2toks.append(b2tok)
        bert_batch_after_padding, bert_lens = \
            pad_trunc_batch(bert_batch_before_padding, max_len=max_bert_len, bert=True)
        # print(bert_batch_after_padding)
        bert2tokens_padded, _ = pad_trunc_batch(bert2toks, max_len=max_bert_len, bert=True, b2t=True)
        bert_batch_ids = torch.LongTensor([self.bert_tokenizer.convert_tokens_to_ids(sent) for \
                                           sent in bert_batch_after_padding])
        bert_seq_ids = torch.LongTensor([[0 for i in range(len(bert_batch_after_padding[0]))] \
                                         for j in range(len(bert_batch_after_padding))])
        # dep_rels = torch.tensor([])
        # dep_inds = torch.tensor([])
        data = torch.tensor(lens), masks, tok_inds, ner_inds, bert_batch_ids, bert_seq_ids, torch.tensor(
            bert2tokens_padded, dtype=torch.long), torch.stack(cap_types)
        return tokens, bert_batch_after_padding, data


if __name__ == "__main__":
    data_path = 'toy_ner_data.tsv'
    reader = DataReader(data_path, "NER")
    # print(sum(map(len,reader.dataset))/reader.data_len)
    # batched_dataset, sentence_lens = group_into_batch(reader.dataset,batch_size = 300)
