from transformers import *

import numpy as np
from dcg_metrics import ndcg_score
import torch
import random
import time
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

cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))


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
        default='biobert_data/datasets/NER_for_QAS_combinedonly',
        type=str,
        required=False,
        help="The root folder containing all the ner datasets.",
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
        default="biobert_data/BioASQ-training8b/training8b_squadformat_train_factoid.json",
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
        default="biobert_data/BioASQ-training8b/training8b_squadformat_train_yesno.json",
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
        default="biobert_data/BioASQ-training8b/training8b_squadformat_train_list.json",
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
                        default='biobert_data/datasets/NER_for_QAS_combinedonly/All-entities/ent_train.tsv',
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
        # try:
        if self.args.biobert_model_path is not None and not self.args.init_bert:
            print("Trying to load from {} ".format(self.args.biobert_model_path))
            print("Using Biobert Model: {} ".format(self.args.biobert_model_path))

            self.bert_model = BertForPreTraining.from_pretrained(self.args.biobert_model_path,
                                                                 from_tf=True, output_hidden_states=True)
        # except:
        # logging.info("Could not load biobert model loading from {}  ".format(pretrained_bert_name))
        # print("Could not load biobert model loading from {}  ".format(pretrained_bert_name))
        else:
            pretrained_bert_name = self.args.model_name_or_path
            if pretrained_bert_name is None:
                print("BERT model name should not be empty when init_model is given")
            if self.args.mlm:
                self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name, output_hidden_states=True)
            else:
                self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name, output_hidden_states=True)

        # print(self.bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert_out_dim = self.bert_model.bert.encoder.layer[11].output.dense.out_features
        self.args.bert_output_dim = self.bert_out_dim
        print("BERT output dim {}".format(self.bert_out_dim))

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
    s = 10
    print("Number of sentences: {}".format(s))
    sentences = []
    labels = []
    for batch in tqdm(eval_dataloader, desc="Bert vec generation"):
        i = i + 1
        if i > s:
            break
        with torch.no_grad():
            if dataset_type == "qas":
                batch = tuple(t.to(device) for t in batch)
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                bert2toks = batch[-1]
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
            bert_hiddens = similarity._get_bert_batch_hidden(outputs[-1], bert2toks)
            cls_vector = bert_hiddens[:, 0, :]
            # print("CLS vector shape: {}".format(cls_vector.shape))
            dataset_vector.extend(cls_vector.detach().cpu())

    dataset_vectors = torch.stack(dataset_vector)

    dataset_vectors = dataset_vectors.detach().cpu().numpy()
    print("Shape {}".format(dataset_vectors.shape))
    return dataset_vectors, sentences, labels


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
    dataset = DataReader(ner_file_path, "NER", for_eval=True, tokenizer=similarity.bert_tokenizer,
                         batch_size=128, crf=False)
    all_vectors, ner_sentences, ner_labels = get_bert_vectors(similarity, dataset, dataset_type="ner")
    vectors = np.array(all_vectors)
    return vectors, ner_sentences, ner_labels


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


def load_store_qas_vectors():
    args = parse_args()
    vector_folder = args.vector_save_folder
    dataset_name = args.squad_train_factoid_file
    save_path = os.path.split(dataset_name)[0].split("/")[-1] + ".hdf5"
    if os.path.exists(save_path):
        print("Found qas vectors previously stored...")
        vectors = load_vectors(file_path)
        return vectors
    else:
        print("Qas vectors not found...")
        similarity = Similarity()
        vectors = store_qas_vectors(similarity, args)
        return vectors


def train_qas_model():
    qas_vectors = load_store_qas_vectors()
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

    return best_model


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
    limit = 10000
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


def select_ner_subset(vectors, size=500):
    """
        Complete this script to graduate from Ph.D
    :param vectors: list of BERT-based vector representations
    :return:  list of indices of the selected sentences for the given size
    """
    best_model = train_qas_model()
    top_inds = get_topN_similar_single_iterative_penalize(best_model, vectors, size)
    return [0, 2, 4]


def write_subset_dataset(indices, sentences, labels, save_path):
    s = "\n\n".join(["\n".join(["{}\t{}".format(s, l) for s, l in zip(sentences[i], labels[i])]) for i in indices])
    with open(save_path, "w") as o:
        o.write(s)


def store_ner_vectors(similarity, args):
    vectors, sentences, labels = get_ner_vectors(similarity, args)
    print("Final shape of ner vectors: {}".format(vectors.shape))
    vector_folder = args.vector_save_folder
    # dataset_name = args.ner_train_file
    # dataset_name = os.path.split(dataset_name)[0].split("/")[-1] + ".hdf5"
    # if not os.path.exists(vector_folder):
    #     os.makedirs(vector_folder)
    # dataset_path = os.path.join(vector_folder, dataset_name)
    # with h5py.File(dataset_path, "w") as h:
    #     h["vectors"] = np.array(vectors)


def store_ner_subset(similarity, args, save_file_path):
    b = time.time()
    vectors, sentences, labels = get_ner_vectors(similarity, args)
    e = time.time()
    t = round(e - b, 3)
    print("Time to get ner vectors: {}".format(t))
    size = 10
    b = time.time()
    indices = select_ner_subset(vectors, size)
    e = time.time()
    t = round(e - b, 3)
    print("Time to select ner subset of size {}: {}".format(size, t))
    write_subset_dataset(indices, sentences, labels, save_file_path)


def generate_store_ner_subsets():
    args = parse_args()
    similarity = Similarity()
    ner_root_folder = args.ner_root_folder
    ner_datasets = list(filter(lambda x: os.path.isdir(os.path.join(ner_root_folder, x)), os.listdir(ner_root_folder)))
    print("Generate subsets for {} datasets".format(len(ner_datasets)))
    ner_datasets = [os.path.join(ner_root_folder, x) for x in ner_datasets]
    # store_ner_vectors(similarity, args)
    # store_qas_vectors(similarity,args)
    subset_sizes = [10,20]
    for dataset_name in ner_datasets:
        folder_name = os.path.split(dataset_name)[-1]
        print("Generating subsets for {}...".format(folder_name))
        for s in subset_sizes:
            save_folder_path = os.path.join(ner_root_folder,"subset_{}_{}".format(folder_name,s))
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            save_file_path = os.path.join(save_folder_path,"ent_train.tsv")
            train_file_name = os.path.join(dataset_name, "ent_train.tsv")
            args.ner_train_file = train_file_name
            print("NER file: {}".format(train_file_name))
            store_ner_subset(similarity, args,save_file_path)
            file_names = ["ent_devel.tsv", "ent_test.tsv"]
            for file_name in file_names:
                file_path = os.path.join(dataset_name,file_name)
                save_path = os.path.join(save_folder_path,file_name)
                small_dataset = get_small_dataset(file_path,size = 1000)
                with open(save_path,"w") as w:
                    w.write(small_dataset)


def main():
    generate_store_ner_subsets()

if __name__ == "__main__":
    main()
