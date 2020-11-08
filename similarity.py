from transformers import *

import numpy as np
from dcg_metrics import ndcg_score
import torch
import random
import os
import torch.nn as nn
from nerreader import DataReader, ner_document_reader
import argparse
import datetime
from stopwords import english_stopwords
import logging
from numpy import dot
from numpy.linalg import norm
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))

pretrained_bert_name = 'bert-base-cased'
gettime = lambda x=datetime.datetime.now(): "{}_{}_{}_{}".format(x.month, x.day, x.hour, x.minute)

exp_prefix = gettime()
print("Time  {} ".format(exp_prefix))
random_seed = 12345
rng = random.Random(random_seed)
log_path = 'main_logger'
logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler(log_path, 'w', 'utf-8')],
                    format='%(levelname)s - %(message)s')


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_gradient(losses, step_size):
    return (losses[-1] - losses[int(max(0, len(losses) - step_size - 1))]) / step_size


def hugging_parse_args():
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

    parser.add_argument('--ner_train_file', type=str, default='ner_data/all_entities_train_dev.tsv',
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
        self.args = hugging_parse_args()
        self.train_file_name = "ent_train.tsv"
        self.test_file_name = "ent_test.tsv"
        if not os.path.isdir(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.device = self.args.device
        # try:
        if self.args.biobert_model_path is not None and not self.args.init_bert:
            print("Trying to load from {} ".format(self.args.biobert_model_path))
            print("Using Biobert Model{} ".format(self.args.biobert_model_path))

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

    def get_all_dataset_bertvectors(self):
        self.dataset_bert_vectors = [self.get_dataset_bertvector(d) for d in self.ner_readers]

    def get_dataset_bertvector(self, dataset):
        """

        :param dataset: A ner dataset consisting of batched sentences?
        :return: A vector representing the document?
        """
        device = self.args.device
        dataset.for_eval = False
        dataset_vector = []
        for i in range(500):
            with torch.no_grad():
                tokens, bert_batch_after_padding, data = dataset[i]
                data = [d.to(device) for d in data]
                sent_lens, masks, tok_inds, ner_inds, \
                bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = data
                outputs = self.bert_model(bert_batch_ids, token_type_ids=bert_seq_ids)
                bert_hiddens = self._get_bert_batch_hidden(outputs[-1], bert2toks)
                cls_vector = bert_hiddens[0, 0, :]
                dataset_vector.append(cls_vector)
        print("In dataset sentence similarities")
        sims = [cos_sim(dataset_vector[i], dataset_vector[j]) for i in range(len(dataset_vector)) for j in
                range(i + 1, len(dataset_vector))]
        max_sim = max(sims)
        min_sim = min(sims)
        print("{} {}".format(max_sim, min_sim))
        dataset_vector = torch.stack(dataset_vector)
        print("After stack shape : {}".format(dataset_vector.shape))
        dataset_vector = torch.mean(dataset_vector, 0)
        print("Dataset vector shape {}".format(dataset_vector.shape))
        return dataset_vector.detach().cpu().numpy()

    def get_all_tfidf_vector_representations(self):
        if hasattr(self, "docs"):
            vectorizer = TfidfVectorizer(max_df=0.7, stop_words=english_stopwords)
            tf_idf = vectorizer.fit_transform(self.docs)
            feature_names = vectorizer.get_feature_names()
            return tf_idf, feature_names
        else:
            print("Docs must be generated to get tfidf vectors")

    def get_documents(self, sent_len=None):
        train_file_names = self.get_train_file_names()
        self.docs = [ner_document_reader(file, sent_len) for file in train_file_names]

    def get_lda_representations(self):
        """

        :param document_vectors: Train LDA for all datasets
        :return:
        """
        if not hasattr(self, "dataset_bert_vectors"):
            self.get_all_tfidf_vector_representations()
        vecs = np.array(self.dataset_tfidf_vectors)
        print("Shape of dataset vectors : {} ".format(vecs.shape))
        lda = LatentDirichletAllocation(n_components=self.args.lda_topic_num, random_state=0)
        topic_vectors = lda.fit_transform(vecs)
        print(topic_vectors[:2])

    def get_train_file_names(self):
        train_file_name = "ent_train.tsv"
        if self.args.data_folder is not None:
            print("Getting training files from data folder")
            root = self.args.data_folder
            # train_files = [os.path.join(root, x, train_file_name) for x in os.listdir(root) if
            #                os.path.isdir(os.path.join(root, x)) and os.path.exists(
            #                    os.path.join(root, x, train_file_name)) and x != "BC5CDR-disease"]
            train_files = [os.path.join(root, x, train_file_name) for x in os.listdir(root) if
                           os.path.isdir(os.path.join(root, x)) and os.path.exists(
                               os.path.join(root, x, train_file_name))]
            print("Train files : {}".format(train_files))
            self.args.ner_train_files = train_files
        else:
            print("Getting training files from cli")
            train_files = self.args.ner_train_files
        return train_files

    def load_datasets(self):
        train_files = self.get_train_file_names()
        self.ner_readers = []

        for train_file in train_files:
            ner_reader = DataReader(
                train_file, "NER", tokenizer=self.bert_tokenizer,
                batch_size=self.args.batch_size)
            self.ner_readers.append(ner_reader)

    def get_dataset_names(self):
        return [os.path.split(os.path.split(f)[0])[-1] for f in self.args.ner_train_files]


def get_entities_from_tsv_dataset(file_path, tag_type="BIO"):
    sentences = open(file_path).read().split("\n\n")[:-1]
    entities = defaultdict(Counter)
    if tag_type == "BIO":
        for sent in sentences:
            prev_tag = "O"
            curr_entity = ""
            for token in sent.split("\n"):
                word = token.split()[0]
                label = token.split()[-1]
                if label != "O":
                    if label[0] == "I":
                        curr_entity = curr_entity + " " + word
                    elif label[0] == "B":
                        if prev_tag != "O":
                            if len(curr_entity) > 0:
                                entities[prev_tag][curr_entity] += 1
                        prev_tag = label[2:]
                        curr_entity = word
                else:
                    if prev_tag != "O":
                        entities[prev_tag][curr_entity] += 1
                    prev_tag = "O"
                    curr_entity = ""
            if len(curr_entity) > 0 and prev_tag != "O":
                entities[prev_tag][curr_entity] += 1
    return entities


def get_dist(vec_1, vec_2):
    dist = np.linalg.norm(vec_1 - vec_2)
    return dist


def get_nmf_based_similarities(similarity):
    """

    :param similarity:  Similarity object
    :return:
    """
    sent_len = None
    similarity.get_documents(sent_len=sent_len)
    tf_idf, feature_names = similarity.get_all_tfidf_vector_representations()
    print("Tf-idf shape {}".format(tf_idf.shape))
    lda = NMF(n_components=similarity.args.lda_topic_num, random_state=0)
    topic_vectors = lda.fit_transform(tf_idf)
    dataset_names = similarity.get_dataset_names()
    print("Topic vectors shape {}".format(topic_vectors.shape))
    sims = get_all_cos_similarities(topic_vectors)
    return sims, dataset_names


def prepare_similarity_dict(similarities, datasets):
    sim_dict = defaultdict(dict)
    for i, sims in enumerate(similarities):
        for j, sim in enumerate(sims):
            sim_dict[datasets[i]][datasets[j]] = sim
            # sim_dict[datasets[j]][datasets[i]] = sim
    return sim_dict


def prepare_result_dict(results, datasets):
    res_dict = defaultdict(dict)
    for i, res in enumerate(results):
        for j, r in enumerate(res):
            res_dict[datasets[i]][datasets[j]] = r
    return res_dict


def get_bert_based_similarities(similarity):
    """

    :param similarity:  Similarity object
    :return:
    """
    if not hasattr(similarity, "ner_readers"):
        similarity.load_datasets()
    similarity.get_all_dataset_bertvectors()
    dataset_names = similarity.get_dataset_names()

    bert_vectors = similarity.dataset_bert_vectors
    sims = get_all_cos_similarities(bert_vectors)
    return sims, dataset_names


def harmonic_mean(s_1, s_2):
    return 2 * s_1 * s_2 / (s_1 + s_2)


def get_vocab_similarity(v_1, v_2):
    voc_1 = set(v_1.w2ind.keys())
    voc_2 = set(v_2.w2ind.keys())
    inter = voc_1.intersection(voc_2)
    inter_len = len(inter)
    voc_1_len = len(voc_1)
    voc_2_len = len(voc_2)
    return 2 * inter_len / (voc_1_len + voc_2_len)


def get_vocab_occurence_similarity(v_1, v_2):
    voc_1 = set(v_1.w2ind.keys())
    voc_2 = set(v_2.w2ind.keys())
    inter = voc_1.intersection(voc_2)
    inter_len = len(inter)
    voc_1_len = len(voc_1)
    return inter_len / voc_1_len


def get_all_vocab_similarities(datareaders):
    """

    :param datareaders: A list of DataReader objects
    :return: the ratio of shared vocabularies
    """
    vocab_sims = [[0 for _ in range(len(datareaders))] for _ in range(len(datareaders))]
    for i, data1 in enumerate(datareaders):
        vocab_1 = data1.word_vocab
        for j in range(len(datareaders)):
            data2 = datareaders[j]
            vocab_2 = data2.word_vocab
            vocab_sim = get_vocab_occurence_similarity(vocab_1, vocab_2)
            vocab_sims[i][j] = vocab_sim
            # vocab_sims[j][i] = vocab_sim
    return vocab_sims


def get_diff_ratio(entity_set_1, entity_set_2, same_tag=True):
    """
        Currently only counting the occurrence and non-occurrence
        Used for finding the entity-level (not word-level) rate of OOVs
        If same_tag, only checks the inclusion under the same tag. Used for datasets with different label sets
    """
    total = 0
    diff = 0
    not_found_entities = set()
    found_entities = set()
    for entity in entity_set_1.keys():
        for word in entity_set_1[entity]:
            if same_tag:
                if word not in entity_set_2[entity]:
                    diff += entity_set_1[entity][word]
                    not_found_entities.add(word)
                else:
                    found_entities.add(word)
            else:
                if not any([word in entity_set_2[ent] for ent in entity_set_2.keys()]):
                    diff += entity_set_1[entity][word]
                    not_found_entities.add(word)
                else:
                    found_entities.add(word)
            total += entity_set_1[entity][word]
    return diff / total, list(not_found_entities), list(found_entities)


def get_all_cooccur_similarities(train_entities, test_entities):
    """

    :param datareaders: A list of DataReader objects
    :return: the ratio of shared vocabularies
    """
    datasets = list(train_entities.keys())
    train_cooccurences = [[0 for i in range(len(datasets))] for j in range(len(datasets))]
    for i, key in enumerate(datasets):

        for j, aux in enumerate(datasets):
            if key == aux:
                continue
            ratio, not_found, found_ents = get_diff_ratio(test_entities[key], train_entities[aux],
                                                          same_tag=False)
            ratio, len(not_found), len(found_ents)
            train_cooccurences[i][j] = 1 - ratio

    return train_cooccurences, datasets


def get_shared_vocab_similarities(similarity):
    """

    :param similarity:  Similarity object
    :return:
    """
    if not hasattr(similarity, "ner_readers"):
        similarity.load_datasets()
    dataset_names = similarity.get_dataset_names()
    print(dataset_names)
    vocab_sims = get_all_vocab_similarities(similarity.ner_readers)
    return vocab_sims, dataset_names


def get_cooccur_entity_based_similarities(similarity):
    """

    :param similarity:  Similarity object
    :return:
    """
    if not hasattr(similarity, "all_train_entities"):
        similarity.get_entities()

    coocur_sims, dataset_names = get_all_cooccur_similarities(similarity.all_train_entities,
                                                              similarity.all_test_entities)
    return coocur_sims, dataset_names


def mtl_target_aux_table(file, dataset_names=None, type="Mean"):
    print("Getting {} scores".format(type))
    with open(file, "r") as f:
        results = f.read().split("\n")[1:][:-1]
        fields = ["data", "pre", "rec", "f-1"]
        aux_names = [res.split()[0].split("_")[1] for res in results]
        target_names = [res.split()[0].split("_")[-1] for res in results]
        if dataset_names is None:
            dataset_names = list(set(aux_names).union(set(target_names)))
            print("Dataset names")
            print(dataset_names)
        dataset_inds = {d: i for i, d in enumerate(dataset_names)}
        table = [[0 for _ in range(len(dataset_names))] for _ in range(len(dataset_names))]
        for res in results:
            aux = res.split()[0].split("_")[1]
            target = res.split()[0].split("_")[-1]
            if target not in dataset_names or aux not in dataset_names:
                continue
            if type == "Mean":
                f1 = res.split()[-2]
            elif type == "Max":
                f1 = res.split()[-1]
            table[dataset_inds[target]][dataset_inds[aux]] = f1
    return table, dataset_names


def get_top_n_inds(lda_comp, top_n):
    x = list(zip([i for i in range(len(lda_comp))], lda_comp))
    x.sort(key=lambda a: a[1], reverse=True)
    inds, comps = zip(*x)
    return list(inds)[:top_n]


def display_topic_words(lda, feature_names, top_n=20):
    for comp in lda.components_:
        top_n_inds = get_top_n_inds(comp, top_n)
        print("My topic words are : {}".format(" ".join([feature_names[i] for i in top_n_inds])))


def get_all_cos_similarities(vecs):
    sims = [[0 for _ in range(len(vecs))] for _ in range(len(vecs))]
    for i, vec in enumerate(vecs):
        for j in range(i, len(vecs)):
            vec2 = vecs[j]
            sim = cos_sim(vec, vec2)
            sims[i][j] = sim
            sims[j][i] = sim
    return sims


def result_similarity_corr_plot(sim_dict, res_dict, sim_type="", target_tasks=None):
    """

    :param sim_dict: dict of dicts for similarities
    :param res_dict: dict of dicts for results
    :return: Draw a plot and also get correlation coefficients like pearson etc.
    """
    shapes = ['p', 'x', 'h', '*', 's', 'o', '1', '2', '3', 'o', '1', '2', '3']
    colors = ['r', 'b', 'y', 'g', 'k', 'c', 'm', '#008000', '#009000', 'c', 'm', '#008000', '#009000']

    for target in sim_dict.keys():
        if target_tasks is not None:
            if not target in target_tasks:
                continue
        plt.figure()
        result_dict = res_dict[target]
        aux_shapes = {key: shape for key, shape in zip(list(result_dict.keys()), shapes)}
        target_colors = {key: color for key, color in zip(list(result_dict.keys()), colors)}
        results = sorted(result_dict.items(), key=lambda i: i[1])
        similarity_dict = sim_dict[target]
        plotted_res = []
        for i, res in enumerate(results):
            aux_name = res[0]
            # if aux_name == target:
            #     continue
            plotted_res.append(res[1])
            x = similarity_dict[aux_name]
            mark = aux_shapes[aux_name]
            c = target_colors[target]
            plt.title("{} similarity results for {} dataset".format(sim_type, target))
            plt.scatter(x, res[1], marker=mark, color=c,
                        label=aux_name)
            plt.legend(title="Aux. task", loc='upper center', bbox_to_anchor=(0.5, 1.0),
                       ncol=3)
        width = max(plotted_res) - min(plotted_res)
        plt.yticks(plotted_res + [max(plotted_res) + width / 2])
        plt.savefig("res_{}_sim_corrplot_target_{}.png".format(sim_type, target))
        plt.close()


def get_correlations(res_dict, sim_dict, separate=True):
    keys = list(res_dict.keys())
    correlation_stats = {}
    ress = []
    sims = []
    if separate:
        for key in keys:
            aux_names = [aux for aux in res_dict[key].keys()]
            res, sim = [res_dict[key][aux] for aux in aux_names], [sim_dict[key][aux] for aux in aux_names]
            ress.extend(res)
            sims.extend(sim)
            pearson, p_value = pearsonr(res, sim)
            correlation_stats[key] = {"pearson": pearson,
                                      "p_value": p_value}
    pearson, p_p_value = pearsonr(ress, sims)
    spearman, s_p_value = spearmanr(ress, sims)
    correlation_stats["overall"] = {"pearson": pearson,
                                    "pearson_p_value": p_value,
                                    "spearman": spearman,
                                    "spearman_p_value": s_p_value}

    return correlation_stats


def get_similarity_result_correlation(similarity):
    target_tasks = ["BC2GM", "BC4CHEMD"]
    sim_type = "bert" if similarity.args.sim_type is None else similarity.args.sim_type
    if sim_type == "topic":
        sims, dataset_names = get_nmf_based_similarities(similarity)
    elif sim_type == "vocab":
        sims, dataset_names = get_shared_vocab_similarities(similarity)
    elif sim_type == "bert":
        sims, dataset_names = get_bert_based_similarities(similarity)

    mtl_results_file = similarity.args.mtl_results_file
    mtl_table, _ = mtl_target_aux_table(mtl_results_file, dataset_names)
    sim_dict = prepare_similarity_dict(sims, dataset_names)
    results = [[float(r) for r in res] for res in mtl_table]
    res_dict = prepare_result_dict(results, dataset_names)
    result_similarity_corr_plot(sim_dict, res_dict, sim_type=sim_type, target_tasks=target_tasks)
    correlation_stats = get_correlations(res_dict, sim_dict)
    print(correlation_stats["BC2GM"])


def get_best_aux(res_dict, target_tasks=None):
    best_aux_dict = {}
    if target_tasks is None:
        target_tasks = list(res_dict.keys())
    for target in target_tasks:
        best_val = 0
        for key, value in res_dict[target].items():
            if value > best_val and target != key:
                best_aux_dict[target] = key
                best_val = value
    return best_aux_dict


def get_best_aux_by_sim(sim_dict, target_tasks=None):
    best_aux_dict = {}
    if target_tasks is None:
        target_tasks = list(sim_dict.keys())
    for target in target_tasks:
        best_val = 0
        for key, value in sim_dict[target].items():
            if value >= best_val and target != key:
                best_aux_dict[target] = key
                best_val = value
    return best_aux_dict


def get_rank_in_score(score_dict, aux_task, target_task):
    sims = score_dict[target_task]
    my_score = score_dict[target_task][aux_task]
    rank = sum([1 if my_score <= sim and task != target_task else 0 for task, sim in sims.items()])
    return rank


def get_rank_in_sim(sim_dict, aux_task, target_task):
    sims = sim_dict[target_task]
    my_score = sim_dict[target_task][aux_task]
    rank = sum([1 if my_score <= sim and task != target_task else 0 for task, sim in sims.items()])
    return rank


def get_similarity_wrapper(similarity, sim_type):
    if sim_type == "topic":
        sims, dataset_names = get_nmf_based_similarities(similarity)
    elif sim_type == "vocab":
        sims, dataset_names = get_shared_vocab_similarities(similarity)
    elif sim_type == "bert":
        sims, dataset_names = get_bert_based_similarities(similarity)
    elif sim_type == "cooccur":
        sims, dataset_names = get_cooccur_entity_based_similarities(similarity)
    else:
        print("Undefined sim type : {}".format(sim_type))
        return {}
    sim_dict = prepare_similarity_dict(sims, dataset_names)
    if sim_type == "cooccur":
        print(sim_dict)
    return sim_dict


def combine_sims(sim1, sim2):
    combined = defaultdict(dict)
    key_list = list(sim1.keys())
    for key in key_list:
        for key2 in key_list:
            if key == key2:
                continue
            aver = (sim1[key][key2] + sim2[key][key2]) / 2
            # print("Keys {} {} sim {} sim {} combined {}".format(key, key2, sim1[key][key2], sim2[key][key2], aver))
            combined[key][key2] = aver
    return combined


def get_all_sims_dict(similarity):
    sim_types = ["topic", "vocab", "cooccur","bert"]
    combined_all_sims_dict = {}
    all_sims_dict = {}
    for sim_type in sim_types:
        sim_dict = get_similarity_wrapper(similarity, sim_type)
        all_sims_dict[sim_type] = sim_dict
        combined_all_sims_dict[sim_type] = sim_dict
    for i, sim in enumerate(sim_types):
        for j, sim2 in enumerate(sim_types):
            if i >= j:
                continue
            name = "{}_{}".format(sim, sim2)
            print(name)
            combined_all_sims_dict[name] = combine_sims(all_sims_dict[sim], all_sims_dict[sim2])

    return combined_all_sims_dict


def get_result_dict_wrapper(mtl_results_file):
    mtl_table, datasets = mtl_target_aux_table(mtl_results_file)
    results = [[float(r) for r in res] for res in mtl_table]
    res_dict = prepare_result_dict(results, datasets)
    return res_dict


def get_ndcg_score(res_dict, target, sim_dict, sim_type):
    aux_list = [x for x in list(res_dict[target].keys()) if x != target]
    relevances = [res_dict[target][aux] for aux in aux_list]
    scores = [sim_dict[sim_type][target][aux] / 100 for aux in aux_list]
    my_score = ndcg_score(relevances, scores)
    print("NDCG Score of {} for {} is {}".format(sim_type, target, my_score))
    return my_score


def compare_similarity_methods(similarity):
    # target_tasks = ["BC2GM", "BC4CHEMD", "JNLPBA", "BC5CDR-chem", "s800", "linnaeus","BC5CDR-disease","NCBI-disease","conll-eng"]
    target_tasks = None
    sims_dict = get_all_sims_dict(similarity)
    mtl_results_file = similarity.args.mtl_results_file
    res_dict = get_result_dict_wrapper(mtl_results_file)
    best_datasets = get_best_aux(res_dict)
    # best_datasets_by_sim = get_best_aux(sim_dict)
    best_dataset_ranks = defaultdict(list)
    best_aux_pred_ranks = defaultdict(list)
    ndcg_scores = defaultdict(list)
    best_aux_dict_by_sim = {sim_type: get_best_aux_by_sim(sim_dict) for sim_type, sim_dict in sims_dict.items()}
    if target_tasks is None:
        target_tasks = [d for d in best_datasets]
    for target_task in target_tasks:
        best_aux = best_datasets[target_task]
        print("Best aux for {} is {} ".format(target_task, best_aux))
        for sim_type, sim_dict in sims_dict.items():
            best_auxs = best_aux_dict_by_sim[sim_type]
            print(sim_type, best_auxs)
            best_aux_by_sim = best_auxs[target_task]
            print("Best aux according to {} for {} : {}".format(sim_type, target_task, best_aux_by_sim))
            rank_of_best_aux_in_sim = get_rank_in_score(res_dict, best_aux_by_sim, target_task)
            my_rank = get_rank_in_sim(sim_dict, best_aux, target_task)
            print("Bert rank according to {} for {}: {}".format(sim_type, best_aux, my_rank))
            best_dataset_ranks[sim_type].append(my_rank)
            nd_score = get_ndcg_score(res_dict, target_task, sims_dict, sim_type)
            ndcg_scores[sim_type].append(nd_score)
            best_aux_pred_ranks[sim_type].append(rank_of_best_aux_in_sim)
    # for sim_type, sim_dict in sims_dict.items():
    #     print(sim_type)
    #     result_similarity_corr_plot(sim_dict, res_dict, sim_type=sim_type, target_tasks=target_tasks)
    best_dataset_ranks = {x: np.mean(s) for x, s in best_dataset_ranks.items()}
    best_aux_pred_ranks = {x: np.mean(s) for x, s in best_aux_pred_ranks.items()}
    ndcg_scores = {x: np.mean(s) for x, s in ndcg_scores.items()}
    print("All NDCG Scores")
    print(ndcg_scores)

    print("Average NDCG scores")
    pretty_print_dict(ndcg_scores)
    print_latex_dict(["NDCG Scores"], ndcg_scores)

    print("Average rank of best auxiliary task according to similarity measure")
    pretty_print_dict(best_dataset_ranks)
    print_latex_dict(["Average rank of best auxiliary"], best_dataset_ranks)

    print("Average rank of best auxiliary predictions")
    pretty_print_dict(best_aux_pred_ranks)
    print_latex_dict(["Average rank of best auxiliary predictions"], best_aux_pred_ranks)

    combined_dicts = combine_dicts([ndcg_scores, best_dataset_ranks, best_aux_pred_ranks])
    print_latex_dict(["NDCG Scores", "Best auxiliary dataset rank","Rank of best auxiliary prediction"],combined_dicts)


def combine_dicts(dicts):
    return {x: [my_dict[x] for my_dict in dicts] for x in dicts[0].keys()}


def pretty_print_dict(dictionary):
    for k, v in dictionary.items():
        print("{}\t{}".format(k, round(v, 3)))


def print_latex_dict(col_headers, dictionary):
    key = list(dictionary.keys())[0]
    row_num = 1 if type(dictionary[key]) != list else len(dictionary[key])
    # first_line = "\{c\| {} \}\\hline\n".format("".join(["c" for c in range(row_num)]))
    header = "&" + "&".join([h.replace("_","-") for h in col_headers]) + "\\\\\\hline\n"
    rows = ""
    for key in dictionary:
        if type(dictionary[key]) == list:
            row = key + "&" + "&".join([str(round(x, 3)) if type(x) != str else x for x in dictionary[key]]) + "\\\\\n"
        else:
            x = dictionary[key]
            row = key + "&" + str(str(round(x, 3)) if type(x) != str else x) + "\\\\\n"
        rows = rows + row
    table = header + rows
    print(table)


def main():
    similarity = Similarity()
    # get_similarity_result_correlation(similarity)
    # print_latex_dict(["NDCG Score"],{"BC2GM":10,"ARDA":20})
    compare_similarity_methods(similarity)
    # sims, datasetner_document_readers = get_cooccur_entity_based_similarities(similarity)
    # sims, datasets = get_shared_vocab_similarities(similarity)
    # sim_dict = prepare_similarity_dict(sims, datasets)
    # print(sim_dict)
    # topic_sims, topic_dataset_names = get_nmf_based_similarities(similarity)
    # vocab_sims, vocab_dataset_names = get_shared_vocab_similarities(similarity)
    # bert_sims, bert_dataset_names = get_bert_based_similarities(similarity)


if __name__ == "__main__":
    main()
