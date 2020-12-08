from transformers import *

import numpy as np
from dcg_metrics import ndcg_score
import torch
import random
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

    parser.add_argument('--ner_train_file', type=str, default='biobert_data/datasets/NER_for_QAS_combinedonly/All-entities/ent_train.tsv',
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
    eval_dataloader = DataLoader(dataset, batch_size=1) if  dataset_type =="qas" else dataset # Doesnt work with batch_size > 1 atm
    similarity.bert_model = similarity.bert_model.to(device)
    print("Getting bert vectors...")
    i = 0
    for batch in tqdm(eval_dataloader, desc="Bert vec generation"):
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
            outputs = similarity.bert_model(**bert_inputs)
            # print("Output shape: {}".format(outputs[-1][0].shape))
            bert_hiddens = similarity._get_bert_batch_hidden(outputs[-1], bert2toks)
            cls_vector = bert_hiddens[:, 0, :]
            # print("CLS vector shape: {}".format(cls_vector.shape))
            dataset_vector.extend(cls_vector.detach().cpu())

    dataset_vectors = torch.stack(dataset_vector)

    dataset_vectors = dataset_vectors.detach().cpu().numpy()
    print("Shape {}".format(dataset_vectors.shape))
    return dataset_vectors


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
        vectors = get_bert_vectors(similarity, dataset)
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


def get_ner_vectors(similarity, args):
    ner_file_path = args.ner_train_file
    dataset = DataReader(ner_file_path, "NER", for_eval=True,tokenizer=similarity.bert_tokenizer,
                         batch_size=128, crf=False)
    all_vectors = get_bert_vectors(similarity, dataset, dataset_type="ner")
    vectors = np.array(all_vectors)
    return vectors

def store_ner_vectors(similarity, args):
    vectors = get_ner_vectors(similarity, args)
    print("Final shape of ner vectors: {}".format(vectors.shape))
    vector_folder = args.vector_save_folder
    dataset_name = args.ner_train_file
    dataset_name = os.path.split(dataset_name)[0].split("/")[-1] + ".hdf5"
    if not os.path.exists(vector_folder):
        os.makedirs(vector_folder)
    dataset_path = os.path.join(vector_folder, dataset_name)
    with h5py.File(dataset_path, "w") as h:
        h["vectors"] = np.array(vectors)

def main():
    args = parse_args()
    similarity = Similarity()
    store_ner_vectors(similarity, args)
    # store_qas_vectors(similarity,args)


if __name__ == "__main__":
    main()
