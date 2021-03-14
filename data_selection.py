from transformers import *

import numpy as np
from dcg_metrics import ndcg_score
import torch
import random
import time
import json
import os
import torch.nn as nn
from tqdm import tqdm, trange
from nerreader import DataReader, ner_document_reader
import argparse
import datetime
from stopwords import english_stopwords
import logging
from numpy import dot
import h5py
from numpy.linalg import norm
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.spatial import distance

from generate_subsets import get_small_dataset
from functools import reduce
import numpy as np

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from reader import TrainingInstance, BertPretrainReader, MyTextDataset, mask_tokens, pubmed_files, \
    squad_load_and_cache_examples

pretrained_bert_name = 'bert-base-cased'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))


def cos_sim(a, b):
    if type(a) == np.ndarray:
        return dot(a, b) / (norm(a) * norm(b))
    elif type(a) == torch.Tensor:
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def get_vocab_similarity(voc_1, voc_2):
    inter = voc_1.intersection(voc_2)
    inter_len = len(inter)
    voc_1_len = len(voc_1)
    voc_2_len = len(voc_2)
    return 2 * inter_len / (voc_1_len + voc_2_len)


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--load_model_path", default="key_models/mybiobert_finetunedsquad", type=str, required=False,
        help="The path to load the model to continue training."
    )
    parser.add_argument(
        "--crf", default=False, action="store_true",
        help="Whether to use CRF for NER head"
    )
    parser.add_argument(
        "--only_lr_curve", default=False, action="store_true",
        help="If set, skips the evaluation, only for NER task"
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        default=True,
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )

    parser.add_argument(
        "--sim_type",
        type=str,
        help="Similarity type",
    )

    parser.add_argument(
        "--ner_result_file",
        default='ner_results',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--ner_root_folder",
        default='biobert_data/datasets/NER_1303/All-entities',
        type=str,
        required=False,
        help="The root folder containing all the ner datasets.",
    )
    parser.add_argument(
        "--save_root_folder",
        default='biobert_data/datasets/subsets_NER_1303/',
        type=str,
        required=False,
        help="The root folder containing all the ner datasets.",
    )
    parser.add_argument(
        "--save_folder_pref",
        default='subset',
        type=str,
        required=False,
        help="Prefix of the folders for all the subsets!",
    )

    parser.add_argument(
        "--vector_save_folder",
        default='bert_vectors',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--qas_result_file",
        default='qas_results',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_dir",
        default='save_dir',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--target_index",
        type=int,
        default=-1,
        required=False,
        help="The index of the target ner task (inside the eval files)",
    )
    parser.add_argument(
        "--ner_train_files",
        default=["a", "b"],
        nargs="*",
        type=str,
        required=False,
        help="The list of training files for the ner task",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=False,
        help="The folder containing all datasets",
    )
    parser.add_argument(
        "--mtl_results_file",
        default="multiner_results_2107",
        type=str,
        required=False,
        help="The folder containing all datasets",
    )

    parser.add_argument(
        "--ner_test_files",
        default=["a", "b"],
        nargs="*",
        type=str,
        required=False,
        help="The list of test files for the ner task",
    )
    parser.add_argument(
        "--model_type", type=str, default='bert', required=False,
        help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=pretrained_bert_name,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        default=True,
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--squad_dir",
        default='.',
        type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument(
        "--cache_folder",
        default="data_cache",
        type=str,
        help="Directory to cache the datasets",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--example_num",
        default=0,
        type=int,
        help="Number of examples to train the data"
    )
    parser.add_argument(
        "--biobert_tf_config",
        default="../biobert_data/biobert_v1.1_pubmed/bert_config.json",
        type=str,
        help="Index file for biobert pretrained model"
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--biobert_tf_model",
        default="../biobert_data/biobert_v1.1_pubmed/model.ckpt-1000000.index",
        type=str,
        help="Index file for biobert pretrained model"
    )
    parser.add_argument(
        "--only_squad",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--qa_type",
        default=None,
        choices=['list', 'yesno', 'factoid']
    )
    parser.add_argument(
        "--squad_yes_no",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--squad_train_factoid_file",
        default="biobert_data/BioASQ-training9b/training9b_squadformat_train_factoid.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--squad_predict_factoid_file",
        default="biobert_data/BioASQ-6b/train/Snippet-as-is/BioASQ-train-factoid-6b-snippet.json",
        type=str,
        help="the input evaluation file. if a data dir is specified, will look for the file there"
             + "if no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--squad_train_yesno_file",
        default="biobert_data/BioASQ-training9b/training9b_squadformat_train_yesno.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--squad_predict_yesno_file",
        default="biobert_data/BioASQ-6b/train/Snippet-as-is/BioASQ-train-yesno-6b-snippet.json",
        type=str,
        help="the input evaluation file. if a data dir is specified, will look for the file there"
             + "if no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--squad_train_list_file",
        default="biobert_data/BioASQ-training9b/training9b_squadformat_train_list.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--squad_predict_list_file",
        default="biobert_data/BioASQ-6b/train/Snippet-as-is/BioASQ-train-list-6b-snippet.json",
        type=str,
        help="the input evaluation file. if a data dir is specified, will look for the file there"
             + "if no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--squad_train_file",
        default="train-v1.1.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--squad_predict_file",
        default="dev-v1.1.json",
        type=str,
        help="the input evaluation file. if a data dir is specified, will look for the file there"
             + "if no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="when splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=128,
        type=int,
        help="the maximum number of tokens for the question. questions longer than this will "
             "be truncated to this length.",
    )

    parser.add_argument(
        "--biobert_model_path",
        default="biobert_data/biobert_v1.1_pubmed",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--biobert_model_name",
        default="dmis-lab/biobert-v1.1",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--init_bert",
        default=False,
        action="store_true",
        help="If invoked, initializes the model from bert instead of biobert",
    )
    parser.add_argument(
        "--load_model",
        default=True,
        action="store_false",
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--model_save_name", default=None, type=str, help="Model name to save"
    )
    parser.add_argument(
        "--mode", default="qas", choices=['qas', 'multiner', 'joint_flat', 'ner', 'qas_ner'],
        help="Determine in which mode to use the Multi-tasking framework"
    )
    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--lda_topic_num", type=int, default=5, help="Number of topics for lda"
    )
    parser.add_argument(
        "--tfidf_dim", type=int, default=4000, help="Number of tfidf dim"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--predict", default=False, action="store_true", help="Whether to run prediction only")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--total_train_steps", default=-1, required=False, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete \
        the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix\
         as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        "--n_best_size", default=20, type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument('--ner_train_file', type=str,
                        default='biobert_data/datasets/NER_1303/All-entities/ent_train.tsv',
                        help='training file for ner')
    parser.add_argument('--ner_dev_file', type=str, default='ner_data/all_entities_test.tsv',
                        help='development file for ner')
    parser.add_argument('--ner_test_file', type=str, default='ner_data/all_entities_test.tsv', help='test file for ner')
    parser.add_argument('--ner_vocab_path', type=str, default='ner_vocab', help='training file for ner')
    # parser.add_argument("--warmup_steps", default=5, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--batch_size', type=int, default=1, help='NER Batch size token based (not sentence)')
    args = parser.parse_args()
    args.device = device
    return args


class Similarity(nn.Module):
    def __init__(self):
        super(Similarity, self).__init__()
        self.args = parse_args()
        self.train_file_name = "ent_train.tsv"
        self.test_file_name = "ent_test.tsv"
        if not os.path.isdir(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.device = self.args.device
        if self.args.biobert_model_path is not None and not self.args.init_bert:
            print("Trying to load from: {}".format(self.args.biobert_model_name))
            self.bert_model = BertForTokenClassification.from_pretrained(self.args.biobert_model_name,
                                                                         output_hidden_states=True)
            self.bert_model.classifier = nn.Identity()
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.biobert_model_name)
        else:
            pretrained_bert_name = self.args.model_name_or_path
            if pretrained_bert_name is None:
                print("BERT model name should not be empty when init_model is given")
            if self.args.mlm:
                self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name, output_hidden_states=True)
            else:
                self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name, output_hidden_states=True)
            self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

        self.bert_out_dim = 768
        self.number_of_subsets = 10
        self.args.bert_output_dim = self.bert_out_dim
        print("BERT output dim {}".format(self.bert_out_dim))

        print("Generating qas vectors...")
        qas_vectors = self.load_store_qas_vectors()
        self.qas_vectors = qas_vectors
        qas_vocab = get_qas_vocab(self.args)
        print("QAS vocab contains {} words".format(len(qas_vocab)))
        self.qas_vocab = qas_vocab

    def _get_bert_batch_hidden(self, hiddens, bert2toks, layers=[-2, -3, -4]):
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]), 0)
        batch_my_hiddens = []
        for means, bert2tok in zip(meanss, bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i, b2t in enumerate(bert2tok):
                if i > 0 and b2t != bert2tok[i - 1]:
                    my_hiddens.append(torch.mean(torch.stack(my_token_hids), 0))
                    # my_token_hids = [means[i+1]]  ## we skip the CLS token
                    my_token_hids = [means[i]]  ## Now I dont skip the CLS token
                else:
                    # my_token_hids.append(means[i+1])
                    my_token_hids = [means[i]]  ## Now I dont skip the CLS token
            my_hiddens.append(torch.mean(torch.stack(my_token_hids), 0))
            sent_hiddens = torch.stack(my_hiddens)
            batch_my_hiddens.append(sent_hiddens)
        return torch.stack(batch_my_hiddens)

    def load_store_qas_vectors(self):
        args = parse_args()
        vector_folder = args.vector_save_folder
        dataset_name = args.squad_train_factoid_file
        save_path = os.path.join(vector_folder, os.path.split(dataset_name)[0].split("/")[-1] + ".hdf5")

        if os.path.exists(save_path):
            print("Found qas vectors previously stored...")
            vectors = load_vectors(save_path)
            return vectors
        else:
            print("Qas vectors not found...")
            vectors = store_qas_vectors(self, args)
            return vectors

    def _get_token_to_bert_predictions(self, predictions, bert2toks):
        # logging.info("Predictions shape {}".format(predictions.shape))

        # logging.info("Bert2toks shape {}".format(bert2toks.shape))
        bert_predictions = []
        for pred, b2t in zip(predictions, bert2toks):
            bert_preds = []
            for b in b2t:
                bert_preds.append(pred[b])
            stack = torch.stack(bert_preds)
            bert_predictions.append(stack)
        stackk = torch.stack(bert_predictions)
        return stackk

    def _get_squad_bert_batch_hidden(self, hiddens, layers=[-2, -3, -4]):
        return torch.mean(torch.stack([hiddens[i] for i in layers]), 0)

    def _get_squad_to_ner_bert_batch_hidden(self, hiddens, bert2toks, layers=[-2, -3, -4], device='cpu'):
        pad_size = hiddens[-1].shape[1]
        hidden_dim = hiddens[-1].shape[2]
        pad_vector = torch.tensor([0.0 for i in range(hidden_dim)]).to(device)
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]), 0)
        batch_my_hiddens = []
        batch_lens = []
        for means, bert2tok in zip(meanss, bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i, b2t in enumerate(bert2tok):
                if i > 0 and b2t != bert2tok[i - 1]:
                    my_hiddens.append(torch.mean(torch.stack(my_token_hids), 0))
                    # my_token_hids = [means[i+1]]  ## we skip the CLS token
                    my_token_hids = [means[i]]  ## Now I dont skip the CLS token
                else:
                    # my_token_hids.append(means[i+1])
                    my_token_hids = [means[i]]  ## Now I dont skip the CLS token
            my_hiddens.append(torch.mean(torch.stack(my_token_hids), 0))
            batch_lens.append(len(my_hiddens))
            for i in range(pad_size - len(my_hiddens)):
                my_hiddens.append(pad_vector)
            sent_hiddens = torch.stack(my_hiddens)
            batch_my_hiddens.append(sent_hiddens)
        # for sent_hidden in batch_my_hiddens:
        # logging.info("Squad squeezed sent shape {}".format(sent_hidden.shape))
        return torch.stack(batch_my_hiddens), torch.tensor(batch_lens)

    def get_entities(self):
        data_folder = self.args.data_folder
        train_file_list = self.args.ner_train_files
        test_file_list = self.args.ner_test_files
        train_file_name = self.train_file_name
        test_file_name = self.test_file_name
        all_train_entities = {}
        all_test_entities = {}
        if data_folder is None and (train_file_list is None or test_file_list is None):
            print("At least data_folder or train_file_list/test_file_list must be defined")
        if data_folder is not None:
            datasets = os.listdir(data_folder)
            for d in datasets:
                folder = os.path.join(data_folder, d)
                if not os.path.isdir(folder) or train_file_name not in os.listdir(folder):
                    continue
                entity_type = d
                test_data_path = "{}/{}".format(folder, test_file_name)
                test_entities = get_entities_from_tsv_dataset(test_data_path)
                train_data_path = "{}/{}".format(folder, train_file_name)
                train_entities = get_entities_from_tsv_dataset(train_data_path)
                all_train_entities[entity_type] = train_entities
                all_test_entities[entity_type] = test_entities
        else:
            for tr, ts in zip(train_file_list, test_file_list):
                entity_type = os.path.split(os.path.split(tr)[0])[-1]
                test_data_path = ts
                test_entities = get_entities_from_tsv_dataset(test_data_path)
                train_data_path = tr
                train_entities = get_entities_from_tsv_dataset(train_data_path)
                all_train_entities[entity_type] = train_entities
                all_test_entities[entity_type] = test_entities
        self.all_train_entities = all_train_entities
        self.all_test_entities = all_test_entities


def get_bert_vectors(similarity, dataset, dataset_type="qas"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_vector = []
    eval_dataloader = DataLoader(dataset,
                                 batch_size=1) if dataset_type == "qas" else dataset  # Doesnt work with batch_size > 1 atm
    similarity.bert_model = similarity.bert_model.to(device)
    print("Getting bert vectors...")
    i = 0
    s = len(dataset)
    print("Number of sentences: {}".format(s))
    sentences = []
    labels = []
    for batch in tqdm(eval_dataloader, desc="Bert vec generation"):
        if i >= s:
            break
        i = i + 1
        with torch.no_grad():
            if dataset_type == "qas":
                # batch = tuple(t.to(device) for t in batch)
                # bert_inputs = {
                #     "input_ids": batch[0],
                #     "attention_mask": batch[1],
                #     "token_type_ids": batch[2],
                # }
                bert2toks = batch[-1]

                # do not input padding part!!
                input_ids = []
                attention_mask = []
                token_type_ids = []
                for i, inp_ids in enumerate(batch[0]):
                    pad_length = sum([1 if a else 0 for a in inp_ids == 0])
                    l = len(inp_ids) - pad_length
                    input_ids.append(inp_ids[:l])
                    attention_mask.append(batch[1][i][:l])
                    token_type_ids.append(batch[2][i][:l])
                bert_inputs = {
                    "input_ids": torch.stack(input_ids).to(device),
                    "attention_mask": torch.stack(attention_mask).to(device),
                    "token_type_ids": torch.stack(token_type_ids).to(device),
                }
            elif dataset_type == "ner":
                tokens, bert_batch_after_padding, data = batch
                data = [d.to(device) for d in data]
                sent_lens, masks, tok_inds, ner_inds, \
                bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = data
                bert_inputs = {
                    "input_ids": bert_batch_ids,
                    "token_type_ids": bert_seq_ids
                }
                label_vocab = dataset.label_vocab
                for toks, n_inds in zip(tokens, ner_inds):
                    my_labels = label_vocab.unmap(n_inds)
                    # print("Tokens: {}".format(toks))
                    # print("Labels: {}".format(my_labels))
                    # print("# tokens: {}  # labels: {}".format(len(toks), len(my_labels)))
                    sentence = toks[1:len(toks) - 1]
                    sentences.append(sentence)
                    labels.append(my_labels[1:len(toks) - 1])

            outputs = similarity.bert_model(**bert_inputs)
            # print("Output shape: {}".format(outputs[-1][0].shape))
            layers = [-1]
            bert_hiddens = torch.mean(torch.stack([outputs[-1][i] for i in layers]), 0)
            # bert_hiddens = similarity._get_bert_batch_hidden(outputs[-1], bert2toks)

            # CLS-based approach
            cls_vector = bert_hiddens[:, 0, :]
            # print("CLS vector shape: {}".format(cls_vector.shape))
            dataset_vector.extend(cls_vector.detach().cpu())

            # Mean-based approach
            mean_vector = torch.mean(bert_hiddens[:, :, :], dim=1)

    dataset_vectors = torch.stack(dataset_vector)

    dataset_vectors = dataset_vectors.detach().cpu().numpy()
    print("Shape {}".format(dataset_vectors.shape))
    return dataset_vectors, sentences, labels


def get_qas_vocab(args):
    vocab = set()
    f, l, y = args.squad_train_factoid_file, args.squad_train_list_file, args.squad_train_yesno_file
    root = os.path.split(f)[0]
    vocab_file = os.path.join(root, "vocab.txt")
    if os.path.exists(vocab_file):
        print("Vocab exists in {}!".format(vocab_file))
        vocab = open(vocab_file, "r", encoding="utf-8").read().split("\n")
        vocab = set(vocab)
        if len(vocab) > 100:
            return vocab
    for file in [f, l, y]:
        print("Adding {} vocab...".format(file))
        d = json.load(open(file, "r"))
        for q in d["data"][0]["paragraphs"]:
            for qu in q["qas"]:
                vocab = vocab.union(set([x for x in qu["question"].split()]))
            c = q["context"]
            vocab = vocab.union(set([x for x in c.split()]))
    with open(vocab_file, "w", encoding="utf-8") as w:
        w.write("\n".join(x for x in vocab))
    return vocab


def get_ner_vocab(ner_sentences):
    vocab = set()
    for sent in ner_sentences:
        vocab = vocab.union(set([x for x in sent]))
    return vocab


def get_qas_vectors(similarity, args):
    qas_train_datasets = {}
    q_types = ["list", "yesno", "factoid"]
    all_vectors = []
    for q in q_types:
        dataset, examples, feats = squad_load_and_cache_examples(args,
                                                                 similarity.bert_tokenizer,
                                                                 yes_no=q == "yesno",
                                                                 output_examples=True,
                                                                 type=q,
                                                                 skip_list=[])
        vectors, _, _ = get_bert_vectors(similarity, dataset)
        all_vectors.extend(vectors)
    vectors = np.array(all_vectors)
    return vectors


def store_qas_vectors(similarity, args):
    vectors = get_qas_vectors(similarity, args)
    print("Final shape of qas vectors: {}".format(vectors.shape))
    vector_folder = args.vector_save_folder
    dataset_name = args.squad_train_factoid_file
    dataset_name = os.path.split(dataset_name)[0].split("/")[-1] + ".hdf5"
    if not os.path.exists(vector_folder):
        os.makedirs(vector_folder)
    dataset_path = os.path.join(vector_folder, dataset_name)
    with h5py.File(dataset_path, "w") as h:
        h["vectors"] = np.array(vectors)
    return np.array(vectors)


def load_vectors(file_path, dim=768):
    with h5py.File(file_path, "r") as h:
        vectors = h["vectors"][:]
        vectors = vectors.reshape(-1, dim)
        return vectors


def get_ner_vectors(similarity, args):
    ner_file_path = args.ner_train_file

    if not hasattr(similarity, "ner_sentences"):
        print("Generating ner vectors and sentences")
        dataset = DataReader(ner_file_path, "NER", for_eval=True, tokenizer=similarity.bert_tokenizer,
                             batch_size=256, crf=False)
        all_vectors, ner_sentences, ner_labels = get_bert_vectors(similarity, dataset, dataset_type="ner")
        vectors = np.array(all_vectors)
        similarity.ner_vectors = vectors
        similarity.ner_sentences = ner_sentences
        print(similarity.ner_sentences[0])
        similarity.ner_labels = ner_labels
    else:
        print("Ner vectors are already generated")
        vectors = similarity.ner_vectors
        ner_sentences = similarity.ner_sentences
        ner_labels = similarity.ner_labels
    return similarity, vectors, ner_sentences, ner_labels


def multiple_clusters(vectors, k_start, k_end, algo="kmeans"):
    models = []
    for k in range(k_start, min(len(vectors) + 1, k_end + 1)):
        if algo == "kmeans":
            model = KMeans(n_clusters=k, random_state=0).fit(vectors)
        elif algo == "gm":
            model = GaussianMixture(n_components=k).fit(vectors)
        else:
            print("Unknown algo: {}".format(algo))
            return []
        models.append(model)
    return models


def train_qas_model(similarity):
    if not hasattr(similarity, "qas_vectors"):
        print("Generating qas vectors...")
        qas_vectors = similarity.load_store_qas_vectors()
        similarity.qas_vectors = qas_vectors
    else:
        print("Qas vectors are already generated...")
        qas_vectors = similarity.qas_vectors

    clustering_begin = time.time()
    k_start = 2
    k_finish = 7
    print("Training gm model on qas vectors!")
    models = multiple_clusters(qas_vectors, k_start, k_finish, "gm")

    # Intrinsically Evaluate
    sample_size = 10000
    np.random.shuffle(qas_vectors)
    sample_feats = qas_vectors
    aics = [model.aic(sample_feats) / reduce(lambda x, y: x * y, sample_feats.shape, 1) for model in models]
    best_model_index = np.argmin(aics)
    best_k = k_start + best_model_index
    best_model = models[best_model_index]
    print("Best gm model found with k={}".format(best_k))
    labels = best_model.predict(qas_vectors)
    clust_sizes = Counter(labels)
    print("Size of each cluster: {}".format(clust_sizes))
    clustering_end = time.time()
    clustering_time = round(clustering_end-clustering_begin,3)
    print("{} seconds for qas topic model training".format(clustering_time))
    return best_model, similarity, clust_sizes


def min_mahalanobis_distance(vec_1, model):
    means = model.means_
    precisions = model.precisions_
    min_dist = min([distance.mahalanobis(vec_1, mean, precision) for mean, precision in zip(means, precisions)])
    return min_dist


def mahalanobis_distance(vec_1, vec_2, inv):
    dist = distance.mahalanobis(vec_1, vec_2, inv)
    return dist


def get_penalty(v, selected_vectors, precision):
    penalty = sum([mahalanobis_distance(v, v2, precision) for v2 in selected_vectors])
    penalty = penalty / len(selected_vectors)
    return penalty


def get_topN_similar_single(target_model, source_vectors, N):
    """
        Gets the top N inds for each cluster
    """
    top_inds = {}
    l = 0
    used_indices = []
    for mean, prec in zip(target_model.means_, target_model.precisions_):

        mah_dists = [mahalanobis_distance(v, mean, prec) for v in source_vectors]
        zipped = list(zip([i for i in range(len(mah_dists))], mah_dists))
        zipped.sort(key=lambda x: x[1])
        indices, dists = list(zip(*zipped))
        print("{} indices {} dists N: {}".format(len(indices), len(dists), N))
        my_inds = []
        i = 0
        while len(my_inds) < N and i < len(indices):
            index = indices[i]
            if index not in used_indices:
                my_inds.append(index)
            i = i + 1
        print("Length of my indices: {}".format(len(my_inds)))
        my_inds.sort()
        # print("My indices: {}".format(my_inds))
        used_indices.extend(my_inds)
        top_inds[l] = my_inds
        l = l + 1
    for l in top_inds:
        for l2 in top_inds:
            if l == l2:
                continue
            for index in top_inds[l]:
                if index in top_inds[l2]:
                    print("{} is included in lists for  both {} and {}.".format(index, l, l2))
    return top_inds


def get_topN_similar_single_iterative_penalize(target_model, source_vectors, N):
    """
        Selects top N sentences for each topic inside the model
        Force model to select diverse examples
    """
    top_inds = {}
    l = 0
    used_indices = []
    for mean, prec in zip(target_model.means_, target_model.precisions_):
        my_inds = get_topN_withpenalty(source_vectors, mean, prec, N, used_indices)
        used_indices.extend(my_inds)
        top_inds[l] = my_inds
        l = l + 1
    return top_inds


def get_topN_withpenalty(vectors, mean, precision, N, skip_list):
    a = 0.8
    b = 1 - a
    limit = 1000
    print("Selecting {} vectors from {} vectors".format(N, len(vectors)))
    mah_dists = [mahalanobis_distance(v, mean, precision) if i not in skip_list else 999999 for i, v in
                 enumerate(vectors)]
    zipped = list(zip([i for i in range(len(mah_dists))], mah_dists))
    zipped.sort(key=lambda x: x[1])
    indices, dists = list(zip(*zipped))
    indices = indices[:limit]
    index = indices[0]
    selected_vectors = [vectors[index]]
    my_inds = [index]
    remaining_indices = set(indices)
    remaining_indices = list(remaining_indices.difference(set(my_inds)))
    while len(my_inds) < N:
        scores = {r: a * mah_dists[r] - 10 * b * get_penalty(vectors[r], selected_vectors, precision) for r in
                  remaining_indices}
        index_score_tuples = [(r, scores[r]) for r in remaining_indices]
        index_score_tuples.sort(key=lambda x: x[1])
        best_index = index_score_tuples[0][0]
        my_inds.append(best_index)
        selected_vectors.append(vectors[best_index])
        remaining_indices = set(remaining_indices)
        remaining_indices = list(remaining_indices.difference(set(my_inds)))
    print("{} vectors {} indices".format(len(selected_vectors), len(my_inds)))
    return my_inds


def topic_instance_based_selection(similarity, vectors, sizes):
    if not hasattr(similarity, "qas_model"):
        best_model, similarity, clust_sizes = train_qas_model(similarity)
        similarity.qas_model = best_model
        similarity.clust_sizes = clust_sizes
    else:
        best_model = similarity.qas_model
        clust_sizes = similarity.clust_sizes
    num_clusters = len(best_model.means_)

    print("Best model has {} clusters".format(num_clusters))
    print("Clust sizes: {}".format(clust_sizes))
    max_size = max(sizes)
    top_inds = get_topN_similar_single(best_model, vectors, max_size)

    total_size = sum(clust_sizes.values())
    all_inds_dict = {}
    for size in sizes:
        all_inds = []
        for k, v in top_inds.items():
            ratio = clust_sizes[k] / total_size
            s = int(size * ratio)
            print("{} indices for {}. Clust size: {}. Top {} will be added..".format(len(v), k, clust_sizes[k], s))
            all_inds.extend(v[:s])
        all_inds.sort()
        print("Top 5 inds", all_inds[:5])
        all_inds_dict[size] = all_inds
    return all_inds_dict, similarity


def get_average_similarity_score(source_vectors, target_vectors):
    b = time.time()
    ## Move to GPU for faster computation
    if not type(source_vectors) == torch.Tensor:
        source_vectors = torch.tensor(source_vectors)
    if not type(target_vectors) == torch.Tensor:
        target_vectors = torch.tensor(target_vectors)
    target_vectors = target_vectors.to(DEVICE)
    source_vectors = source_vectors.to(DEVICE)

    source_similarities = [max([cos_sim(s, t) for t in target_vectors]) for i, s in enumerate(source_vectors)]
    for i in tqdm(range(len(source_vectors))):
        s = source_vectors[i]
        max_sim = max([cos_sim(s, t) for t in target_vectors])
        source_similarities.append(max_sim)
    print("Length of source similarities: {}\n\n".format(len(source_similarities)))
    e = time.time()
    t = round(e - b, 3)
    print("{} seconds for average sim score".format(t))
    return sum(source_similarities).detach().cpu().item() / len(source_similarities)


def get_topN_cossimilar(source_vectors, target_vectors, max_size):
    b = time.time()
    source_similarities = []
    for i in tqdm(range(len(source_vectors)), desc="TopN cossimilar"):
        s = source_vectors[i]
        source_similarities.append((max([cos_sim(s, t) for t in target_vectors]).detach().cpu().item(), i))
    source_similarities.sort(key=lambda x: x[0], reverse=True)
    e = time.time()
    t = round(e - b, 3)
    print("{} seconds for topN cos similar...".format(t))
    return [x[-1] for x in source_similarities[:max_size]]


def bert_instance_based_selection(similarity, vectors, sizes):
    print("bert-instance based selection")

    qas_vectors = similarity.qas_vectors

    max_size = max(sizes)

    ## Move to GPU for faster computation
    if not type(qas_vectors) == torch.Tensor:
        qas_vectors = torch.tensor(qas_vectors)
    if not type(vectors) == torch.Tensor:
        vectors = torch.tensor(vectors)
    qas_vectors = qas_vectors.to(similarity.device)
    vectors = vectors.to(similarity.device)
    top_inds = get_topN_cossimilar(qas_vectors, vectors, max_size)
    all_inds_dict = {}
    for size in sizes:
        all_inds_dict[size] = top_inds[:size]
    return all_inds_dict, similarity


def bert_subset_based_selection(similarity, vectors, sizes):
    print("bert-subset based selection")

    number_of_subsets = similarity.number_of_subsets
    indices = [i for i in range(len(vectors))]

    subset_indices = {}
    for s in sizes:
        my_indices = []
        for _ in range(number_of_subsets):
            np.random.shuffle(indices)
            my_indices.append(indices[:s])
        subset_indices[s] = my_indices
        print("{} subsets of size {}".format(len(my_indices), s))

    qas_vectors = similarity.qas_vectors
    all_inds_dict = {}
    for size in sizes:
        best_subset_index = np.argmax(
            [get_average_similarity_score([vectors[i] for i in indices], qas_vectors) for indices in
             subset_indices[size]])
        print("Best index for {}: {} size : {}".format(size, best_subset_index,
                                                       len(subset_indices[size][best_subset_index])))
        all_inds_dict[size] = subset_indices[size][best_subset_index]

    return all_inds_dict, similarity


def select_ner_subsets(similarity, vectors, sizes, method_name="topic-instance"):
    """
        Complete this script to graduate from Ph.D
    :param vectors: list of BERT-based vector representations
    :return:  list of indices of the selected sentences for the given size
    """
    if method_name == "topic-instance":
        all_inds_dict, similarity = topic_instance_based_selection(similarity, vectors, sizes)
    elif method_name == "bert-instance":
        all_inds_dict, similarity = bert_instance_based_selection(similarity, vectors, sizes)
    elif method_name == "bert-subset":
        all_inds_dict, similarity = bert_subset_based_selection(similarity, vectors, sizes)
    elif method_name == "random":
        indices = [i for i in range(len(vectors))]
        all_inds_dict = {}
        for size in sizes:
            np.random.shuffle(indices)
            all_inds_dict[size] = indices[:size]
    return all_inds_dict, similarity


def write_subset_dataset(indices, sentences, labels, save_path):
    # Sometimes writes [SEP] at the end!!
    s = "\n\n".join(
        ["\n".join(["{}\t{}".format(s, l) for s, l in zip(sentences[i], labels[i]) if l != ["[SEP]"]]) for i in
         indices])
    with open(save_path, "w") as o:
        o.write(s)


def store_ner_vectors(similarity, args):
    similarity, vectors, sentences, labels = get_ner_vectors(similarity, args)
    print("Final shape of ner vectors: {}".format(vectors.shape))
    vector_folder = args.vector_save_folder
    dataset_name = args.ner_train_file
    dataset_name = os.path.split(dataset_name)[0].split("/")[-1] + ".hdf5"
    if not os.path.exists(vector_folder):
        os.makedirs(vector_folder)
    dataset_path = os.path.join(vector_folder, dataset_name)
    print("Saving features to: {}".format(dataset_path))
    with h5py.File(dataset_path, "w") as h:
        h["vectors"] = np.array(vectors)


def get_bert_similarity(source_vectors, target_vectors):
    """
        To speed-up calculation, we select 100 vectors (T) from target randomly and get max([cos_sim(s,t') for t' in T])
    """
    sims = []
    sample_size = 100
    for i in tqdm(range(len(source_vectors)), desc="Bert Similarity"):
        s = source_vectors[i]
        np.random.shuffle(target_vectors)
        my_sim = max([cos_sim(s, t) for t in target_vectors[:sample_size]]).detach().cpu().item()
        sims.append(my_sim)
    return sum(sims) / len(sims)


def get_dataset_similarity_scores(similarity, ner_sentences, ner_vectors):
    """

        Shared Voc. Bert/BioBERT-based similarity
        Topic distribution similarity??
        Coccurring entity ratio??
    :param similarity:
    :param ner_sentences:
    :return:
    """

    # Vocab similarity
    ner_vocab = get_ner_vocab(ner_sentences)
    print("NER dataset contains {} words".format(len(ner_vocab)))
    vocab_sim = get_vocab_similarity(similarity.qas_vocab, ner_vocab)
    print("Vocab similarity: {}".format(vocab_sim))

    # Bert-based similarity
    b = time.time()
    print("Getting BERT similarity")
    target_vectors = torch.tensor(similarity.qas_vectors)
    target_vectors = target_vectors.to(similarity.device)
    ner_vectors = torch.tensor(ner_vectors)
    ner_vectors = ner_vectors.to(similarity.device)
    bert_sim = get_bert_similarity(ner_vectors, target_vectors)
    print("BERT similarity: {}".format(bert_sim))
    e = time.time()
    t = round(e - b, 3)
    print("Bert similarity calculated in {} seconds...".format(t))

    sim_scores = {"vocab_similarity": vocab_sim,
                  "bert_similarity": bert_sim}
    return sim_scores


def store_ner_subsets(similarity, args, sizes, save_folder, ner_dataset_name, method_name="topic-instance"):
    save_folder_paths = {size: os.path.join(save_folder, "{}_{}_{}".format(method_name, size, ner_dataset_name))
                         for size in sizes}

    b = time.time()
    similarity, vectors, sentences, labels = get_ner_vectors(similarity, args)
    print("Number of sentences: {}".format(len(sentences)))
    print("Shape of vectors: {}".format(vectors.shape))
    logging.info("Number of sentences: {}".format(len(sentences)))
    logging.info("Shape of vectors: {}".format(vectors.shape))

    e = time.time()
    t = round(e - b, 3)
    logging.info("Time to get ner vectors: {}".format(t))
    print("Time to get ner vectors: {}".format(t))

    b = time.time()
    indices_dict, similarity = select_ner_subsets(similarity, vectors, sizes, method_name=method_name)
    logging.info("Selected {} indices in total.".format(len(indices_dict)))
    print("Selected {} indices in total.".format(len(indices_dict)))

    e = time.time()
    t = round(e - b, 3)
    logging.info("Time to select ner subsets of sizes {}: {}".format(sizes, t))
    print("Time to select ner subsets of sizes {}: {}".format(sizes, t))
    for size, indices in indices_dict.items():
        save_file_path = os.path.join(save_folder_paths[size], "ent_train.tsv")
        ner_sentences = [sentences[i] for i in indices]
        ner_vectors = [vectors[i] for i in indices]
        print("{} ner vectors and {} ner sentences...".format(len(ner_vectors), len(ner_sentences)))
        # similarity scores
        sim_scores = get_dataset_similarity_scores(similarity, ner_sentences, ner_vectors)

        # write subset dataset
        write_subset_dataset(indices, sentences, labels, save_file_path)

        # store ner vectors of this subset
        exp_name = os.path.split(save_folder_paths[size])[-1]
        vector_file_path = os.path.join(save_folder_paths[size], "{}.hdf5".format(exp_name))
        my_vectors = np.array([vectors[i] for i in indices])
        logging.info("My vectors shape: {}".format(my_vectors.shape))
        print("My vectors shape: {}".format(my_vectors.shape))
        with h5py.File(vector_file_path, "w") as h:
            h["vectors"] = my_vectors

        ## Store metadata
        metadata_path = os.path.join(save_folder_paths[size], "metadata.json")
        metadata = {"indices": indices,
                    "method_name": method_name,
                    "folder_name": save_folder_paths[size],
                    "similarity_scores": sim_scores}
        with open(metadata_path, "w") as w:
            json.dump(metadata, w)
    return similarity


def store_ner_folder_vectors():
    args = parse_args()
    ner_folder = args.ner_root_folder
    for ner_dataset in os.listdir(ner_folder):
        p = os.path.join(ner_folder, ner_dataset)
        args.ner_train_file = os.path.join(p, "ent_train.tsv")
        similarity = Similarity()
        store_ner_vectors(similarity, args)


def store_vectors():
    args = parse_args()
    similarity = Similarity()
    store_qas_vectors(similarity, args)
    # store_ner_folder_vectors()


def generate_store_ner_subsets_single(similarity, args, save_folder,
                                      ner_dataset_name, ner_dataset_folder,
                                      sizes, method_name):
    save_folder_paths = [os.path.join(save_folder, "{}_{}_{}".format(method_name, size, ner_dataset_name)) for size in
                         sizes]
    for f in save_folder_paths:
        if not os.path.exists(f):
            os.makedirs(f)
    logging.info("\n\n=== GENERATING SUBSETS FOR {} ===\n\n".format(method_name))
    similarity = store_ner_subsets(similarity, args, sizes, save_folder, ner_dataset_name, method_name=method_name)
    file_names = ["ent_devel.tsv", "ent_test.tsv"]
    for size, f in zip(sizes, save_folder_paths):
        if not os.path.exists(f):
            os.makedirs(f)
        for file_name in file_names:
            file_path = os.path.join(ner_dataset_folder, file_name)
            save_path = os.path.join(f, file_name)
            small_dataset = get_small_dataset(file_path, size=size)
            with open(save_path, "w") as w:
                w.write(small_dataset)


def generate_store_ner_subsets():
    args = parse_args()
    similarity = Similarity()
    ner_root_folder = args.ner_root_folder
    save_root_folder = args.save_root_folder
    # save_root_folder = os.path.split(ner_root_folder)[0]

    # ner_datasets = list(filter(lambda x: os.path.isdir(os.path.join(ner_root_folder, x)), os.listdir(ner_root_folder)))
    # ner_datasets = [os.path.join(ner_root_folder, x) for x in ner_datasets]
    ner_datasets = [ner_root_folder]
    print("Generate subsets for {} datasets...".format(len(ner_datasets)))
    subset_sizes = [1000, 2000, 5000, 10000, 20000]
    # subset_sizes = [10,20,30]
    # methods = ["random","topic-instance", "bert-instance", "bert-subset"]
    methods = ["random"]
    for dataset_name in ner_datasets:
        folder_name = os.path.split(dataset_name)[-1]
        for method in methods:
            print("Generating subsets for {} with {} ...".format(folder_name, method))
            args.ner_train_file = os.path.join(ner_root_folder, "ent_train.tsv")
            generate_store_ner_subsets_single(similarity, args,
                                              save_root_folder,
                                              folder_name,
                                              ner_root_folder,
                                              subset_sizes,
                                              method)


def main():
    generate_store_ner_subsets()


    # args = parse_args()
    # similarity = Similarity()
    # store_ner_vectors(similarity, args)
    # store_vectors()
    # store_ner_folder_vectors()


if __name__ == "__main__":
    main()
