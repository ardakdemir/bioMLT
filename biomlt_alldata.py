from transformers import *

from transformers import get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    # compute_predictions_logits,
    squad_evaluate,
)
from squad_metrics import compute_predictions_logits
import numpy as np
from conll_eval import evaluate_conll_file
from vocab import Vocab
import json
import copy
import torch
import torchvision
import random
import os
import torch.nn as nn
import torch.optim as optim
from reader import TrainingInstance, BertPretrainReader, MyTextDataset, mask_tokens, pubmed_files, \
    squad_load_and_cache_examples
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
import tokenization
from utils import *
from nerreader import DataReader
from nermodel import NerModel
from qasmodel import QasModel
import argparse
from torch.nn import CrossEntropyLoss, MSELoss
import datetime

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
        "--version_2_with_negative",
        action="store_true",
        default=True,
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--config_file",
        default='biomlt_config',
        type=str,
        required=False,
        help="Configuration file for the biomlt model.",
    )
    parser.add_argument(
        "--pred_path",
        default=None,
        type=str,
        required=False,
        help="The output path for storing nbest predictions. Used for evaluating with the bioasq scripts",
    )
    parser.add_argument(
        "--nbest_path",
        default=None,
        type=str,
        required=False,
        help="The output path for storing nbest predictions. Used for evaluating with the bioasq scripts",
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
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=pretrained_bert_name,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
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
        "--mode", default="qas", choices=['qas', 'joint_flat', 'ner', 'qas_ner'],
        help="Determine in which mode to use the Multi-tasking framework"
    )
    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
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

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument('--ner_train_file', type=str, default='ner_data/all_entities_train_dev.tsv',
                        help='training file for ner')
    parser.add_argument('--ner_dev_file', type=str, default='ner_data/all_entities_test.tsv',
                        help='development file for ner')
    parser.add_argument('--ner_test_file', type=str, default='ner_data/all_entities_test.tsv', help='test file for ner')
    parser.add_argument('--ner_vocab_path', type=str, default='ner_vocab', help='training file for ner')

    parser.add_argument('--ner_lr', type=float, default=0.0015, help='Learning rate for ner lstm')
    parser.add_argument("--qas_lr", default=5e-6, type=float, help="The initial learning rate for Qas.")
    parser.add_argument("--qas_adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument('--qas_out_dim', type=int, default=2, help='Output dimension for question answering head')

    # parser.add_argument("--warmup_steps", default=5, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--t_total", default=5000, type=int, help="Total number of training steps")
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--ner_batch_size',
                        type=int, default=50, help='NER Batch size token based (not sentence)')
    parser.add_argument('--eval_batch_size', type=int, default=12, help='Batch size')
    # parser.add_argument('--block_size', type=int, default=128, help='Block size')
    # parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs')

    args = parser.parse_args()
    args.device = device
    return args


def parse_args():
    args = vars(parser.parse_args())
    args['device'] = device
    return args


def generate_pred_content(tokens, preds, truths=None, lens=None, label_voc=None):
    ## this is where the start token and  end token get eliminated!!
    sents = []
    if truths:
        for sent_len, sent, pred, truth in zip(lens, tokens, preds, truths):
            s_ind = 1
            e_ind = 1
            l = list(zip(sent[s_ind:sent_len - e_ind], truth[s_ind:sent_len - e_ind], pred[s_ind:sent_len - e_ind]))
            sents.append(l)
    else:
        for sent, pred in zip(tokens, preds):
            end_ind = -1
            s_ind = 1
            sents.append(list(zip(sent[s_ind:end_ind], pred[s_ind:end_ind])))

    return sents


class BioMLT(nn.Module):
    def __init__(self):
        super(BioMLT, self).__init__()
        self.args = hugging_parse_args()
        if not os.path.isdir(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.device = self.args.device
        # try:
        if self.args.biobert_model_path is not None and not self.args.init_bert:
            print("Trying to load from {} ".format(self.args.biobert_model_path))
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

        self.ner_path = self.args.ner_train_file

        # self.ner_reader = DataReader(self.ner_path, "NER",tokenizer=self.bert_tokenizer,batch_size = 30)
        # self.args.ner_label_vocab = self.ner_reader.label_voc
        # self.ner_head = NerModel(self.args)

        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        self.yesno_head = nn.Linear(self.bert_out_dim, 2)
        # self.qa_outputs =
        self.yesno_soft = nn.Softmax(dim=1)
        self.yesno_loss = CrossEntropyLoss()
        self.yesno_lr = self.args.qas_lr
        self.yesno_optimizer = optim.AdamW([{"params": self.yesno_head.parameters()},
                                            {"params": self.yesno_soft.parameters()}],
                                           lr=self.yesno_lr, eps=self.args.qas_adam_epsilon)

        self.qas_head = QasModel(self.args)

        self.bert_optimizer = AdamW(optimizer_grouped_parameters,
                                    lr=2e-5)
        self.bert_scheduler = get_linear_schedule_with_warmup(
            self.bert_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.t_total
        )
        if self.args.load_model:
            print("Model parameters is loaded from {} ".format(self.args.load_model_path))
            self.load_all_model(self.args.load_model_path)

    # Loads model with missing parameters or extra parameters!!!
    # Solves the previous issue we had for pyJNERDEP
    def load_all_model(self, load_path):
        # self.jointmodel=JointModel(self.args)
        load_path = self.args.load_model_path
        # save_path = os.path.join(self.args['save_dir'],self.args['save_name'])
        logging.info("Model loaded  from: %s" % load_path)
        loaded_params = torch.load(load_path, map_location=torch.device('cpu'))
        my_dict = self.state_dict()
        print("Yes-no head weights before loading")
        before = self.yesno_head.weight[:10]
        print(before)
        pretrained_dict = {k: v for k, v in loaded_params.items() if k in self.state_dict()}
        my_dict.update(pretrained_dict)
        self.load_state_dict(my_dict)
        print("Yes-no head weights after loading")
        print(self.yesno_head.weight[:3])
        # if self.args.mode in ["ner","joint_flat"]:
        #    print("Trying to load ner vocab from {}".format(self.args.ner_vocab_path))
        #    with open(self.args.ner_vocab_path,'r') as np:
        #        ner_w2ind = json.load(np)
        #        self.ner_label_vocab = Vocab(ner_w2ind)

    def save_all_model(self, save_path=None, weights=True):
        if self.args.model_save_name is None and save_path is None:
            save_name = os.path.join(self.args.output_dir, "{}_{}".format(self.args.mode, exp_prefix))
        else:
            if save_path is None:
                save_path = self.args.model_save_name
            save_name = os.path.join(self.args.output_dir, save_path)
        save_dir = os.path.split(save_name)[0]
        if not os.path.isdir(save_dir):
            logging.info("Creating save directory {}".format(save_dir))
            os.makedirs(save_dir)
        if weights:
            logging.info("Saving biomlt model to {}".format(save_name))
            torch.save(self.state_dict(), save_name)
        config_path = os.path.join(self.args.output_dir, self.args.config_file)
        arg = copy.deepcopy(self.args)
        del arg.device
        if hasattr(arg, "ner_label_vocab"):
            if self.args.ner_vocab_path is not None:
                save_path = self.args.ner_vocab_path
            else:
                save_path = "{}_ner_vocab.json".format(exp_prefix)
            print("Saving ner vocab to {} ".format(save_path))
            with open(save_path, "w") as np:
                json.dump(arg.ner_label_vocab.w2ind, np)
            del arg.ner_label_vocab
        arg = vars(arg)
        with open(config_path, 'w') as outfile:
            json.dump(arg, outfile)

    ## We are now averaging over the bert layer outputs for the NER task
    ## We may want to do this for QAS as well?
    ## This is very slow, right?
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

    def load_model(self):
        if self.args.mlm:
            logging.info("Attempting to load  model from {}".format(self.args.output_dir))
            self.bert_model = BertForMaskedLM.from_pretrained(self.args.output_dir)
        else:
            self.bert_model = BertForPreTraining.from_pretrained(self.args.output_dir)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.output_dir)
        sch_path = os.path.join(self.args.output_dir, "scheduler.pt")
        opt_path = os.path.join(self.args.outpt_dir, "optimizer.pt")
        if os.path.isfile(sch_path) and os.path.isfile(opt_path):
            self.bert_optimizer.load_state_dict(torch.load(opt_path, map_location=torch.device('cpu')))
            self.bert_scheduler.load_state_dict(torch.load(sch_path, map_location=torch.device('cpu')))
        logging.info("Could not load model from {}".format(self.args.output_dir))
        logging.info("Initializing Masked LM from {} ".format(pretrained_bert_name))
        # self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name)
        # self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name)

    def forward(self):
        return 0

    def save_model(self):
        out_dir = self.args.output_dir
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print("Saving model checkpoint to {}".format(out_dir))
        logger.info("Saving model checkpoint to {}".format(out_dir))
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.bert_model
        model_to_save.save_pretrained(out_dir)
        self.bert_tokenizer.save_pretrained(out_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(out_dir, "training_args.bin"))
        torch.save(self.bert_optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
        torch.save(self.bert_scheduler.state_dict(), os.path.join(out_dir, "scheduler.pt"))

    def evaluate_qas(self, ind, only_preds=False, types=['factoid', 'list', 'yesno']):
        device = self.args.device
        self.device = device
        args = self.args
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.yesno_head.to(device)
        f1s, totals, exacts = {}, {}, {}
        nbests, preds = {}, {}
        self.bert_model.eval()
        self.qas_head.eval()
        self.yesno_head.eval()
        if self.args.model_save_name is None:
            prefix = gettime() + "_" + str(ind)
        else:
            prefix = self.args.model_save_name
        for type in types:
            print("Evaluation for {} ".format(type))
            if args.pred_path is not None:
                print("Pred path prefix : {}".format(args.pred_path))
            if args.nbest_path is not None:
                print("Nbest path suffix : {}".format(args.nbest_path))
            qas_eval_dataset = self.qas_eval_datasets[type]
            examples = self.qas_eval_examples[type]
            features = self.qas_eval_features[type]
            print("Size of the test dataset {}".format(len(qas_eval_dataset)))
            eval_sampler = SequentialSampler(qas_eval_dataset)
            eval_dataloader = DataLoader(qas_eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
            logger.info("Evaluation {} started for {} type questions ".format(ind, type))
            logger.info("***** Running evaluation {} with only_preds = {}*****".format(prefix, only_preds))
            logger.info("  Num examples = %d", len(qas_eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_results = []
            for batch in tqdm(eval_dataloader, desc="Evaluating"):

                batch = tuple(t.to(self.device) for t in batch)
                # print("Batch shape  {}".format(batch[0].shape))
                # if len(batch[0].shape)==1:
                #    batch = tuple(t.unsqueeze_(0) for t in batch)
                # logging.info(batch[0])
                squad_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    # "start_positions": batch[3],
                    # "end_positions": batch[4],
                }
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                with torch.no_grad():
                    outputs = self.bert_model(**bert_inputs)
                    # squad_inputs["bert_outputs"] = outputs[-1][-2]

                    bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                    # logging.info("Bert out shape {}".format(bert_out.shape))
                    qas_out = self.get_qas(bert_out,
                                           batch,
                                           eval=True,
                                           is_yes_no=self.args.squad_yes_no, type=type)
                    example_indices = batch[3]
                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    if type == 'yesno':
                        output = qas_out[i, :].detach().cpu().numpy()
                        yesno_logit = output
                        # print("What is start_logit {}".format(yesno_logit))
                        probs = self.yesno_soft(torch.tensor(yesno_logit).unsqueeze(0))
                        # print("Yes-no probs : {}".format(probs))
                        result = SquadResult(unique_id,
                                             float(yesno_logit[0]), float(yesno_logit[1]))
                    else:
                        output = [to_list(output[i]) for output in qas_out]
                        start_logit, end_logit = output
                        result = SquadResult(unique_id, start_logit, end_logit)
                    # print(result.start_logits)
                    all_results.append(result)

            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            if args.pred_path is not None:
                output_prediction_file = args.pred_path + "_{}.json".format(type)
            else:
                output_prediction_file = os.path.join(args.output_dir, "{}_predictions_{}.json".format(type, prefix))
            if args.nbest_path is not None:
                output_nbest_file = args.nbest_path + "_{}.json".format(type)
            else:
                output_nbest_file = os.path.join(args.output_dir, "{}_nbest_predictions_{}.json".format(type, prefix))

            output_null_log_odds_file = os.path.join(args.output_dir, "{}_null_odds_{}.json".format(type, prefix))
            print("Length of predictions {} feats  {} examples {} ".format(len(all_results), len(examples),
                                                                           len(features)))
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                args.verbose_logging,
                args.version_2_with_negative,
                args.null_score_diff_threshold,
                self.bert_tokenizer,
                is_yes_no=True if type == "yesno" else False
            )

            if only_preds:
                nbests[type] = output_nbest_file
                preds[type] = output_prediction_file
                continue
                # return output_nbest_file, output_prediction_file
            print("example answer:: ")
            print(examples[0].answers)
            results = squad_evaluate(examples, predictions)
            f1 = results['f1']
            exact = results['exact']
            total = results['total']
            print("RESULTS for {} : f1 {}  exact {} total {} ".format(type, f1, exact, total))
            logging.info("RESULTS for {}: f1 {} exact {} total {} ".format(type, f1, exact, total))
            f1s[type] = f1
            exacts[type] = exact
            totals[type] = total
        if only_preds:
            return nbests, preds
        return f1s, exacts, totals

    def predict_qas(self, batch):
        ## batch_size = 1
        if len(batch[0].shape) == 1:
            batch = tuple(t.unsqueeze_(0) for t in batch)

        ## Not sure if these update in-place?>?>?> Have to check betweenn pytorch versions
        self.bert_model.eval()
        self.qas_head.eval()
        squad_inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        bert_inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        with torch.no_grad():
            outputs = self.bert_model(**bert_inputs)
            squad_inputs["bert_outputs"] = outputs[-1][-2]
            start_pred, end_pred = self.qas_head.predict(**squad_inputs)
            length = torch.sum(batch[1])
            tokens = self.bert_tokenizer.convert_ids_to_tokens(batch[0].squeeze(0).detach().cpu().numpy()[:length])
        logging.info("Example {}".format(tokens))
        logging.info("Answer {}".format(tokens[start_pred:end_pred + 1]))

    def get_qas(self, bert_output, batch, eval=False, is_yes_no=False, type='factoid'):

        # batch = tuple(t.unsqueeze_(0) for t in batch)
        if eval:

            squad_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                'bert_outputs': bert_output
                # "start_positions": batch[3],
                # "end_positions": batch[4],
            }
        else:

            squad_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "bert_outputs": bert_output
            }

        if type in ['factoid', 'list']:
            qas_outputs = self.qas_head(**squad_inputs)
        elif type == 'yesno':
            ##!!!  CLS TOKEN  !!! ##
            yes_no_logits = self.yesno_head(bert_output[:, 0])
            if not eval:
                loss = self.yesno_loss(yes_no_logits, batch[3])
                return (loss, yes_no_logits)
            return yes_no_logits
        # print(qas_outputs[0].item())
        # qas_outputs[0].backward()
        # self.bert_optimizer.step()
        # self.qas_head.optimizer.step()
        return qas_outputs

    def get_ner(self, bert_output, bert2toks, ner_inds=None, predict=False):
        bert_hiddens = self._get_bert_batch_hidden(bert_output, bert2toks)

        if predict:
            all_preds = []
            out_logits = self.ner_head(bert_hiddens, ner_inds, pred=predict)
            preds = []
            voc_size = len(self.ner_reader.label_vocab)
            if self.args.crf:
                sent_len = out_logits.shape[1]
                for i in range(out_logits.shape[0]):
                    pred, score = self.ner_head._viterbi_decode(out_logits[i, :], sent_len)
                    preds.append(pred)

                for pred in preds:
                    ## MAP [CLS] and [SEP] predictions to O
                    print("Before Viterbi result {}".format(pred))
                    # pred = [p // voc_size for p in pred]
                    # print("Before Viterbi result {}".format(pred))
                    pred = list(map(lambda x: "O" if (x == "[SEP]" or x == "[CLS]" or x == "[PAD]") else x,
                                    self.ner_reader.label_vocab.unmap(pred)))

                    all_preds.append(pred)
            else:
                preds = torch.argmax(out_logits, dim=2).detach().cpu().numpy()
                # print("Preds ", preds)
                for pred in preds:
                    ## MAP [CLS] and [SEP] predictions to O
                    p = list(map(lambda x: "O" if (x == "[SEP]" or x == "[CLS]" or x == "[PAD]") else x,
                                 self.ner_reader.label_vocab.unmap(pred)))
                all_preds.append(p)
            all_ner_inds = []
            if ner_inds is not None:
                if not self.args.crf:
                    ner_inds = ner_inds.detach().cpu().numpy()
                else:
                    ner_inds = ner_inds.detach().cpu().numpy() // voc_size
                for n in ner_inds:
                    n_n = list(map(lambda x: "O" if (x == "[SEP]" or x == "[CLS]" or x == "[PAD]") else x,
                                   self.ner_reader.label_vocab.unmap(n)))
                    all_ner_inds.append(n_n)
                return all_preds, all_ner_inds
            else:
                return all_preds
        # logging.info("NER output {} ".format(ner_outs.))
        else:
            ner_outs = self.ner_head(bert_hiddens, ner_inds)
            return ner_outs

    def run_test(self):
        # assert self.args.load_model_path is not None, "Model path to be loaded must be defined to run in predict mode!!!"
        # self.load_all_model(self.args.load_model_path)

        device = self.args.device
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.yesno_head.to(device)
        # self.load_eval_data()
        type = "yesno" if self.args.squad_yes_no else "factoid"
        if self.args.only_squad:
            type = "squad"

        types = self.qas_eval_datasets.keys()
        nbest_files, pred_files = self.evaluate_qas(0, only_preds=True, types=types)
        if self.args.mode in ["ner", "joint_flat"]:
            self.eval_ner()
        for q_t in types:
            print(
                "=== {} question results=== \n Predictions  are saved to {} \n N-best predictions are saved to {} ".format(
                    q_t, pred_files[q_t], nbest_files[q_t]))

    def load_ner_data(self, eval_file=None):
        # now initializing Ner head here !!
        print("Reading NER data from {}".format(self.ner_path))
        self.ner_reader = DataReader(
            self.ner_path, "NER", tokenizer=self.bert_tokenizer,
            batch_size=self.args.ner_batch_size, crf=self.args.crf)
        if self.args.load_model:
            # self.ner_reader.label_vocab = self.ner_label_vocab
            # self.args.ner_label_vocab = self.ner_label_vocab
            print("Model lodaded")
        self.args.ner_label_vocab = self.ner_reader.label_vocab
        # with open(self.args.ner_vocab_path,"w") as np:
        #    json.dump(self.args.ner_label_vocab.w2ind,np)
        print("NER label vocab indexes from training set : {}".format(self.args.ner_label_vocab.w2ind))
        print("Reading NER eval data from {}".format(self.args.ner_dev_file))
        eval_file_path = self.args.ner_dev_file if eval_file is None else eval_file
        self.eval_file = eval_file_path
        print("Will evaluate NER model on {}".format(eval_file_path))
        self.ner_eval_reader = DataReader(
            eval_file_path, "NER", tokenizer=self.bert_tokenizer,
            batch_size=self.args.ner_batch_size, for_eval=True, crf=self.args.crf)
        print("NER label vocab indexes for evaluation set before assignment {} ".format(
            self.ner_eval_reader.label_vocab.w2ind))
        self.ner_eval_reader.label_vocab = self.args.ner_label_vocab
        print("NER label vocab indexes for evaluation set after assignment {} ".format(
            self.ner_eval_reader.label_vocab.w2ind))

    ## training a flat model (multi-task learning hard-sharing)
    def train_qas_ner(self):
        self.load_ner_data()
        eval_file_name = self.eval_file.split("/")[-1].split(".")[0]
        ner_model_save_name = "best_ner_{}_model".format(eval_file_name)
        self.ner_head = NerModel(self.args)
        # type = "yesno" if self.args.squad_yes_no else "factoid"
        # print("Type {}".format(type))
        qa_types = ["yesno", "list", "factoid"]
        device = self.args.device
        # device = "cpu"
        args = hugging_parse_args()
        args.train_batch_size = self.args.batch_size
        self.load_qas_data(args, qa_types=qa_types)
        if self.args.only_squad:
            type = "squad"
        self.device = device
        train_datasets = self.qas_train_datasets
        # train_dataset = ConcatDataset([train_datasets[x] for x in train_datasets])

        if 'factoid' in qa_types or 'list' in qa_types:
            f_l = []
            if 'list' in qa_types:
                f_l.append('list')
            if 'factoid' in qa_types:
                f_l.append('factoid')
            train_dataset = ConcatDataset([train_datasets[x] for x in f_l])
            if "yesno" in qa_types:
                yesno_train_dataset = self.qas_train_datasets["yesno"]
                yesno_sampler = RandomSampler(yesno_train_dataset)
                yesno_dataloader = DataLoader(yesno_train_dataset,
                                              sampler=yesno_sampler, batch_size=args.train_batch_size)
        else:
            train_dataset = self.qas_train_datasets["yesno"]
        # Added here for reproductibility
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        t_totals = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": self.qas_head.parameters(), "weight_decay": 0.0}]
        # self.bert_squad_optimizer =AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        ## Scheduler for sub-components
        # scheduler = get_linear_schedule_with_warmup(
        # self.bert_squad_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_totals)
        tr_loss, logging_loss = 0.0, 0.0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        best_result = 0
        best_ner_f1 = 0
        ner_results = []
        best_sum = 0
        best_result = 0
        best_results = {q: 0 for q in qa_types}
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.yesno_head.to(device)
        self.ner_head.to(device)
        self.yesno_head.to(device)
        self.yesno_soft.to(device)
        self.yesno_loss.to(device)
        self.bert_model.train()
        self.qas_head.train()
        self.ner_head.train()
        self.yesno_head.train()
        self.yesno_loss.train()
        for epoch, _ in enumerate(train_iterator):
            self.bert_model.train()
            self.qas_head.train()
            self.ner_head.train()
            self.yesno_head.train()
            self.yesno_soft.train()
            epoch_loss = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            yesno_epoch_iterator = tqdm(yesno_dataloader, desc="yesno - Iteration")
            yes_size = len(yesno_train_dataset)
            fact_size = len(train_dataset)
            yes_rat = yes_size / (yes_size + fact_size)
            step_len = min(yes_size, fact_size)
            # step_len = 10
            for step, (batch_1, batch_2) in enumerate(zip(epoch_iterator, yesno_epoch_iterator)):
                if step >= step_len or step > len(self.ner_reader):
                    break
                rand = np.random.rand()
                # print("rand val : {} ".format(rand))
                if rand < yes_rat:
                    batch = batch_2
                    type = "yesno"
                else:
                    batch = batch_1
                    type = "factoid"
                # print("BATCH")
                # print(batch)
                # if step >10:
                #    break
                self.bert_optimizer.zero_grad()
                self.ner_head.optimizer.zero_grad()
                self.qas_head.optimizer.zero_grad()
                self.yesno_optimizer.zero_grad()
                # batch = train_dataset[0]
                # batch = tuple(t.unsqueeze(0) for t in batch)
                # logging.info(batch[-1])
                # logging.info(self.bert_tokenizer.convert_ids_to_tokens(batch[0][0].detach().numpy()))
                batch = tuple(t.to(device) for t in batch)
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                # logging.info("Input ids shape : {}".format(batch[0].shape))
                # bert2toks = batch[-1]
                outputs = self.bert_model(**bert_inputs)
                # bert_outs_for_ner , lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1],batch[-1],device=device)
                # print("BERT OUTS FOR NER {}".format(bert_outs_for_ner.shape))
                # ner_outs = self.ner_head(bert_outs_for_ner)
                # ner_outs_2= self.get_ner(outputs[-1], bert2toks)
                # ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs,batch[-1])
                # logging.info("NER OUTS FOR QAS {}".format(ner_outs_for_qas.shape))
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                # logging.info("Bert out shape {}".format(bert_out.shape))
                qas_outputs = self.get_qas(bert_out, batch, eval=False, is_yes_no=self.args.squad_yes_no, type=type)
                qas_loss = qas_outputs[0]
                # empty gradients
                # self.bert_optimizer.zero_grad()
                # self.qas_head.optimizer.zero_grad()
                # self.ner_head.optimizer.zero_grad()

                # get batches for each task to GPU/CPU
                tokens, bert_batch_after_padding, data = self.ner_reader[step]

                data = [d.to(device) for d in data]
                sent_lens, masks, tok_inds, ner_inds, \
                bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = data
                batch = tuple(t.to(device) for t in batch)
                # logging.info(self.bert_tokenizer.convert_ids_to_tokens(batch[0][0].detach().numpy()))
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                # logging.info("Number of sentences in the ner batch : {}".format(len(tokens)))
                # logging.info("Number of sentences in the dep batch : {}".format(batch[0].shape[0]))

                # ner output
                outputs = self.bert_model(bert_batch_ids, token_type_ids=bert_seq_ids)
                # bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
                # loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
                ner_loss, ner_out_logits = self.get_ner(outputs[-1], bert2toks, ner_inds)
                # logging.info("NER out shape : {}".format(ner_out_logits.shape))

                # for hierarchical setting
                # bert_outs_for_ner, lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1], batch[-1],device=device)
                # ner_outs = self.ner_head(bert_outs_for_ner)
                # ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs, batch[-1])

                # qas output
                # logging.info("QAS out shape : {}".format(qas_outputs[1].shape))
                # qas_outputs = self.qas_head(**squad_inputs)
                # print(qas_outputs[0].item())

                # sum losses for backprop
                total_loss = ner_loss + qas_loss
                # total_loss = ner_loss
                total_loss.backward()
                # logging.info("Total loss {}".format(total_loss.item()))
                # logging.info("Total loss {} ner: {}  asq : {}".format(total_loss.item(),
                #                                                       ner_loss.item(),qas_loss.item()))

                # backpropagation
                self.ner_head.optimizer.step()
                self.bert_optimizer.step()
                self.yesno_optimizer.step()
                # not sure if optimizer and scheduler works simultaneously
                # self.bert_scheduler.step()
                self.qas_head.optimizer.step()

                epoch_loss += total_loss.item()
                # if step % 100 == 99:
                #     if self.args.model_save_name is None:
                #         checkpoint_name = self.args.mode + "_" + exp_prefix + "_check_{}_{}".format(epoch, step)
                #     else:
                #         checkpoint_name = self.args.model_save_name + "_check_{}_{}".format(epoch, step)
                #     logging.info("Saving checkpoint to {}".format(checkpoint_name))
                #     self.save_all_model(checkpoint_name)
                #     logging.info("Average loss after {} steps : {}".format(step + 1, epoch_loss / (step + 1)))

            self.bert_model.eval()
            self.qas_head.eval()
            self.ner_head.eval()
            self.yesno_head.eval()
            self.yesno_loss.eval()
            print("Epoch {} is finished, moving to evaluation ".format(epoch))
            with torch.no_grad():
                f1, p, r = self.eval_ner()
                ner_results.append([f1, p, r])
                if f1 > best_ner_f1:
                    best_ner_f1 = f1
                    best_ner_ind = epoch
                    self.save_all_model(ner_model_save_name)
                    print("BEst ner result {}".format(best_ner_f1))
                f1s, exacts, totals = self.evaluate_qas(epoch, types=qa_types)
                # yes_f1, yes_exact, yes_total  = self.evaluate_qas(epoch,type='yesno')
                if self.args.squad_yes_no:
                    print("Yes results {} {} {} ".format(f1, exact, total))

                print("Sum of all f1s {} ".format(sum(f1s.values())))
                print("Sum of best results {} ".format(sum(best_results.values())))
                if sum(f1s.values()) > best_sum:
                    best_sum = sum(f1s.values())
                    print("Overall best model found!! Saving to {} ".format(self.args.model_save_name))
                    save_name = "mode_{}_exp_{}".format(self.args.mode,
                                                        exp_prefix) if self.args.model_save_name is None else self.args.model_save_name
                    self.save_all_model(save_name)
                for q in qa_types:
                    print("Results for {}  f1 : {} exact : {} total : {} ".format(q, f1s[q], exacts[q], totals[q]))
                    f1 = f1s[q]
                    if f1 >= best_results[q]:
                        best_results[q] = f1
                        print("Best f1 of {} for {} ".format(f1, q))
                        save_name = "mode_{}_exp_{}_qtype_{}".format(self.args.mode, exp_prefix,
                                                                     q) if self.args.model_save_name is None else self.args.model_save_name + "_{}".format(
                            q)
                        print("Saving best model for {} questions with {} f1  to {}".format(q, f1,
                                                                                            save_name))
                        logging.info("Saving best model for {} questions with {} f1  to {}".format(q, f1,
                                                                                                   save_name))
                        self.save_all_model(save_name)

        result_save_path = os.path.join(args.output_dir, args.ner_result_file)
        eval_file_name = self.eval_file.split("/")[-1].split(".")[0]
        print("Saving ner results to {} ".format(result_save_path))
        self.write_ner_result(result_save_path, eval_file_name, ner_results, best_ner_ind)

        qas_save_path = os.path.join(self.args.output_dir, self.args.qas_result_file)
        print("Writing results to {}".format(qas_save_path))
        with open(qas_save_path, "a") as out:
            s = "List\tyes-no\tfactoid\n"
            s = s + "\t".join([str(best_results[q]) for q in ["list", "yesno", "factoid"]]) + "\n"
            out.write(s)

    def load_ner_vocab(self):
        if self.args.ner_vocab_path is None:
            print("Ner vocab must be defined for the prediction mode!! ! ! ! !")
            return
        with open(self.args.ner_vocab_path, 'r') as js:
            ner_vocab = json.load(js)
            ner_vocab = Vocab(ner_vocab)
            return ner_vocab

    def load_qas_data(self, args, qa_types=['yesno', 'list', 'factoid'], for_pred=False):
        qas_train_datasets = {}
        qas_eval_datasets = {}
        qas_eval_examples = {}
        qas_eval_features = {}
        if 'yesno' in qa_types:
            if not for_pred:
                qas_train_datasets["yesno"] = squad_load_and_cache_examples(args,
                                                                            self.bert_tokenizer, yes_no=True,
                                                                            type='yesno')
            qas_eval_datasets['yesno'], qas_eval_examples['yesno'], qas_eval_features[
                'yesno'] = squad_load_and_cache_examples(args, self.bert_tokenizer, evaluate=True, output_examples=True,
                                                         yes_no=True, type='yesno')
        if 'list' in qa_types:
            if not for_pred:
                qas_train_datasets["list"] = squad_load_and_cache_examples(args,
                                                                           self.bert_tokenizer, yes_no=False,
                                                                           type='list')
            qas_eval_datasets['list'], qas_eval_examples['list'], qas_eval_features[
                'list'] = squad_load_and_cache_examples(args, self.bert_tokenizer, evaluate=True, output_examples=True,
                                                        yes_no=False, type='list')
        if 'factoid' in qa_types:
            if not for_pred:
                qas_train_datasets["factoid"] = squad_load_and_cache_examples(args,
                                                                              self.bert_tokenizer, yes_no=False,
                                                                              type='factoid')
            qas_eval_datasets['factoid'], qas_eval_examples['factoid'], qas_eval_features[
                'factoid'] = squad_load_and_cache_examples(args, self.bert_tokenizer, evaluate=True,
                                                           output_examples=True, yes_no=True, type='factoid')
        self.qas_train_datasets = qas_train_datasets
        self.qas_eval_datasets = qas_eval_datasets
        self.qas_eval_examples = qas_eval_examples
        self.qas_eval_features = qas_eval_features

    def load_eval_data(self):
        args = self.args
        type = "yesno" if self.args.squad_yes_no else "factoid"
        if self.args.only_squad:
            type = "squad"
        if self.args.qa_type is not None:
            type = self.args.qa_type
        self.qas_eval_dataset, self.qas_eval_examples, self.qas_eval_features = squad_load_and_cache_examples(args,
                                                                                                              self.bert_tokenizer,
                                                                                                              evaluate=True,
                                                                                                              output_examples=True,
                                                                                                              yes_no=self.args.squad_yes_no,
                                                                                                              type=type)
        self.ner_eval_reader = DataReader(
            self.args.ner_test_file, "NER", tokenizer=self.bert_tokenizer,
            batch_size=self.args.ner_batch_size, crf=self.args.crf)
        ner_label_vocab = self.load_ner_vocab()
        print("Prediction data for NER is loaded from {} ".format(self.args.ner_test_file))
        self.ner_eval_reader.label_vocab = ner_label_vocab

    def predict_ner(self):
        self.eval_ner()

    def train_qas(self):
        qa_types = ["yesno", "list", "factoid"]
        device = self.args.device
        print("My device {}".format(device))
        args = hugging_parse_args()
        args.train_batch_size = self.args.batch_size
        self.load_qas_data(args, qa_types=qa_types)
        print("Is yes no ? {}".format(self.args.squad_yes_no))
        # train_dataset = squad_load_and_cache_examples(args,
        #                                              self.bert_tokenizer,
        #                                              yes_no =self.args.squad_yes_no,type='yesno')
        type = "yesno" if self.args.squad_yes_no else "factoid"
        print("Type {}".format(type))
        if self.args.only_squad:
            type = "squad"
        if self.args.qa_type is not None:
            type = self.args.qa_type
        train_datasets = self.qas_train_datasets
        # train_dataset = ConcatDataset([train_datasets[x] for x in train_datasets])

        if 'factoid' in qa_types or 'list' in qa_types:
            f_l = []
            if 'list' in qa_types:
                f_l.append('list')
            if 'factoid' in qa_types:
                f_l.append('factoid')
            train_dataset = ConcatDataset([train_datasets[x] for x in f_l])
            if "yesno" in qa_types:
                yesno_train_dataset = self.qas_train_datasets["yesno"]
                yesno_sampler = RandomSampler(yesno_train_dataset)
                yesno_dataloader = DataLoader(yesno_train_dataset,
                                              sampler=yesno_sampler, batch_size=args.train_batch_size)
        else:
            train_dataset = self.qas_train_datasets["yesno"]

        # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        t_totals = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": self.qas_head.parameters(), "weight_decay": 0.0}]
        # self.bert_squad_optimizer =AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        ## Scheduler for sub-components
        # scheduler = get_linear_schedule_with_warmup(
        # self.bert_squad_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_totals)
        tr_loss, logging_loss = 0.0, 0.0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        best_result = 0
        best_sum = 0
        best_results = {q: 0 for q in qa_types}
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.yesno_head.to(device)
        self.yesno_loss.to(device)
        # self.ner_head.to(device)
        print("weights before training !!")
        print(self.qas_head.qa_outputs.weight[-10:])
        print("Concat size {} yesno size {}".format(len(train_dataset), len(yesno_train_dataset)))
        for epoch, _ in enumerate(train_iterator):
            total_loss = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            yesno_epoch_iterator = tqdm(yesno_dataloader, desc="yesno - Iteration")
            self.bert_model.train()
            self.qas_head.train()
            self.yesno_head.train()
            self.yesno_loss.train()
            yes_size = len(yesno_train_dataset)
            fact_size = len(train_dataset)
            yes_rat = yes_size / (yes_size + fact_size)
            step_len = min(yes_size, fact_size)
            # step_len = 10
            for step, (batch_1, batch_2) in enumerate(zip(epoch_iterator, yesno_epoch_iterator)):
                if step >= step_len:
                    break
                rand = np.random.rand()
                # print("rand val : {} ".format(rand))
                if rand < yes_rat:
                    batch = batch_2
                    type = "yesno"
                else:
                    batch = batch_1
                    type = "factoid"
                # print("BATCH")
                # print(batch)
                # if step >10:
                #    break
                self.bert_optimizer.zero_grad()
                self.qas_head.optimizer.zero_grad()
                self.yesno_optimizer.zero_grad()
                # batch = train_dataset[0]
                # batch = tuple(t.unsqueeze(0) for t in batch)
                # logging.info(batch[-1])
                # logging.info(self.bert_tokenizer.convert_ids_to_tokens(batch[0][0].detach().numpy()))
                batch = tuple(t.to(device) for t in batch)
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                print("Question type {}".format(type))
                # logging.info("Input ids shape : {}".format(batch[0].shape))
                # bert2toks = batch[-1]
                outputs = self.bert_model(**bert_inputs)
                # bert_outs_for_ner , lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1],batch[-1],device=device)
                # print("BERT OUTS FOR NER {}".format(bert_outs_for_ner.shape))
                # ner_outs = self.ner_head(bert_outs_for_ner)
                # ner_outs_2= self.get_ner(outputs[-1], bert2toks)
                # ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs,batch[-1])
                # logging.info("NER OUTS FOR QAS {}".format(ner_outs_for_qas.shape))
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                # logging.info("Bert out shape {}".format(bert_out.shape))
                qas_outputs = self.get_qas(bert_out, batch, eval=False, is_yes_no=self.args.squad_yes_no, type=type)

                loss = qas_outputs[0]
                loss.backward()
                self.bert_optimizer.step()
                self.qas_head.optimizer.step()
                self.yesno_optimizer.step()
                total_loss += loss.item()
                if step % 100 == 99:
                    print("Loss {} ".format(loss.item()))
                    if self.args.model_save_name is None:
                        checkpoint_name = self.args.mode + "_" + exp_prefix + "_check_{}_{}".format(epoch, step)
                    else:
                        checkpoint_name = self.args.model_save_name + "_check_" + str(step)
                    logging.info("Saving checkpoint to {}".format(checkpoint_name))
                    self.save_all_model(checkpoint_name)
                    logging.info("Average loss after {} steps : {}".format(step + 1, total_loss / (step + 1)))
            print("Total loss {} for epoch {} ".format(total_loss, epoch))
            print("Epoch {} is finished, moving to evaluation ".format(epoch))
            f1s, exacts, totals = self.evaluate_qas(epoch, types=qa_types)
            # yes_f1, yes_exact, yes_total  = self.evaluate_qas(epoch,type='yesno')
            if self.args.squad_yes_no:
                print("Yes results {} {} {} ".format(f1, exact, total))

            print("Sum of all f1s {} ".format(sum(f1s.values())))
            print("Sum of best results {} ".format(sum(best_results.values())))
            if sum(f1s.values()) > best_sum:
                best_sum = sum(f1s.values())
                print("Overall best model found!! Saving to {} ".format(self.args.model_save_name))
                save_name = "mode_{}_exp_{}".format(self.args.mode,
                                                    exp_prefix) if self.args.model_save_name is None else self.args.model_save_name
                self.save_all_model(save_name)
            for q in qa_types:
                print("Results for {}  f1 : {} exact : {} total : {} ".format(q, f1s[q], exacts[q], totals[q]))
                f1 = f1s[q]
                if f1 >= best_results[q]:
                    best_results[q] = f1
                    print("Best f1 of {} for {} ".format(f1, q))
                    save_name = "mode_{}_exp_{}_qtype_{}".format(self.args.mode, exp_prefix,
                                                                 q) if self.args.model_save_name is None else self.args.model_save_name + "_{}".format(
                        q)
                    print("Saving best model for {} questions with {} f1  to {}".format(q, f1,
                                                                                        save_name))
                    logging.info("Saving best model for {} questions with {} f1  to {}".format(q, f1,
                                                                                               save_name))
                    self.save_all_model(save_name)
            qas_save_path = os.path.join(self.args.output_dir, self.args.qas_result_file)
            print("Writing results to {}".format(qas_save_path))
            with open(qas_save_path, "a") as out:
                s = "List\tyes-no\tfactoid\n"
                s = s + "\t".join([str(best_results[q]) for q in ["list", "yesno", "factoid"]]) + "\n"
                out.write(s)

    def pretrain_mlm(self):
        device = self.args.device
        epochs_trained = 0
        epoch_num = self.args.epoch_num
        batch_size = self.args.batch_size
        block_size = self.args.block_size
        huggins_args = hugging_parse_args()

        # file_list = pubmed_files()
        file_list = ["PMC6958785.txt", "PMC6961255.txt"]
        train_dataset = MyTextDataset(self.bert_tokenizer, huggins_args, file_list, block_size=block_size)
        print("Dataset size {} ".format(len(train_dataset)))
        print(train_dataset[0])
        train_sampler = RandomSampler(train_dataset)

        def collate(examples):
            return pad_sequence(examples, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate
        )
        t_totals = len(train_dataloader) // self.args.epoch_num
        # self.dataset = reader.create_training_instances(file_list,bert_tokenizer)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        self.bert_scheduler = get_linear_schedule_with_warmup(
            self.bert_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_totals
        )
        if self.args.load_model:
            print("Model loaded")
            self.bert_load_model()
        print("BERT after loading weights")
        print(self.bert_model.bert.encoder.layer[11].output.dense.weight)
        self.bert_model.to(device)
        self.bert_model.train()
        print("Model is being trained on {} ".format(next(self.bert_model.parameters()).device))
        train_iterator = trange(
            # epochs_trained, int(huggins_args.num_train_epochs), desc="Epoch")
            epochs_trained, int(epoch_num), desc="Epoch")
        # set_seed(args)  # Added here for reproducibility
        for _ in train_iterator:
            for step, batch in enumerate(epoch_iterator):
                # print("Batch shape {} ".format(batch.shape))
                # print("First input {} ".format(batch[0]))
                self.bert_optimizer.zero_grad()  ## update mask_tokens to apply curriculum learnning!!!!
                inputs, labels = mask_tokens(batch, self.bert_tokenizer, huggins_args)
                tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs.cpu().detach().numpy()[0, :])
                label_tokens = self.bert_tokenizer.convert_ids_to_tokens(labels.cpu().detach().numpy()[0, :])
                logging.info("Tokens {}".format(tokens))
                logging.info("Labels ".format(label_tokens))
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.bert_model(inputs, masked_lm_labels=labels)
                loss = outputs[0]
                logging.info("Loss obtained for batch of {} is {} ".format(batch.shape, loss.item()))
                loss.backward()
                self.bert_optimizer.step()
                if step == 2:
                    break
            self.save_model()
            logging.info("Training is finished moving to evaluation")
            self.mlm_evaluate()

    def mlm_evaluate(self, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = out_dir = self.args.output_dir

        # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        eval_batch_size = 1
        file_list = ["PMC6958785.txt"]
        eval_dataset = MyTextDataset(self.bert_tokenizer, self.huggins_args, file_list, block_size=128)

        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        def collate(examples):
            return pad_sequence(examples, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=collate
        )

        # multi-gpu evaluate
        # if args.n_gpu > 1:
        #    model = torch.nn.DataParallel(model)

        # Eval!
        model = self.bert_model
        logger.info("***** Running evaluation on {} *****".format(file_list))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = mask_tokens(batch, self.bert_tokenizer, self.huggins_args)
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)

            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result

    ##parallel reading not implemented for training
    def pretrain(self):
        file_list = ["PMC6961255.txt"]
        reader = BertPretrainReader(file_list, self.bert_tokenizer)
        # dataset = reader.create_training_instances(file_list,self.bert_tokenizer)
        tokens = reader.dataset[1].tokens
        logging.info(tokens)
        input_ids = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)  # Batch size 1
        # token_type_ids= torch.tensor(dataset[1].segment_ids).unsqueeze(0)
        # print(input_ids.shape)
        # print(dataset[1].segment_ids)
        # next_label = torch.tensor([ 0 if dataset[1].is_random_next  else  1])
        token_ids, mask_labels, next_label, token_type_ids = reader[0]
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        for i in range(10):
            self.bert_optimizer.zero_grad()
            # print("Input shape {}".format(token_ids.shape))
            outputs = self.bert_model(token_ids, token_type_ids=token_type_ids)
            prediction_scores, seq_relationship_scores = outputs[:2]
            vocab_dim = prediction_scores.shape[-1]
            masked_lm_loss = loss_fct(prediction_scores.view(-1, vocab_dim), mask_labels.view(-1))
            next_sent_loss = loss_fct(seq_relationship_scores.view(-1, 2), next_label.view(-1))
            loss = masked_lm_loss + next_sent_loss
            loss.backward()
            self.bert_optimizer.step()
        pred_tokens = self.bert_tokenizer.convert_ids_to_tokens(
            torch.argmax(prediction_scores, dim=2).detach().cpu().numpy()[0])
        logging.info("{} {} ".format("Real tokens", tokens))
        logging.info("{} {} ".format("Predictions", pred_tokens))

    def train_ner(self):
        eval_file = self.args.ner_dev_file
        eval_file_name = os.path.split(eval_file)[1].split(".")[0]
        ner_type = os.path.split(eval_file)[0].split("/")[-1]
        model_save_name = "best_ner_model_on_{}".format(ner_type)
        self.load_ner_data(eval_file=eval_file)
        # if self.args.load_model:
        #    self.ner_reader.label_vocab = self.ner_label_vocab
        #    self.args.ner_label_vocab = self.ner_label_vocab
        self.args.ner_label_vocab = self.ner_reader.label_vocab
        self.ner_head = NerModel(self.args)
        device = self.args.device
        # device = "cpu"
        args = hugging_parse_args()
        args.train_batch_size = self.args.batch_size
        print("Starting training for NER in {} ".format(device))
        print("Tokens  : ", self.ner_reader[0][0])
        self.bert_model.to(device)
        self.ner_head.to(device)
        self.bert_model.train()
        self.ner_head.train()
        results = []
        best_epoch = 0
        best_f1 = 0
        avg_ner_losses = []
        epoch_num = args.num_train_epochs
        print("Total epochs over data {} ".format(epoch_num))
        len_data = len(self.ner_reader)
        eval_interval = len_data//2
        # eval_interval = 100
        print("Length of each epoch {}".format(len_data))
        epoch_num = epoch_num * len_data // eval_interval
        print("Will train for {} epochs ".format(epoch_num))

        for j in tqdm(range(epoch_num), desc="Epochs"):
            ner_loss = 0
            self.bert_model.train()
            self.ner_head.train()
            # eval_interval = len(self.ner_reader)
            for i in tqdm(range(eval_interval), desc="Training"):
                self.bert_optimizer.zero_grad()
                self.ner_head.optimizer.zero_grad()
                tokens, bert_batch_after_padding, data = self.ner_reader[i]
                # print("Number of sentences in the batch : {}".format(len(tokens)))
                data = [d.to(device) for d in data]
                sent_lens, masks, tok_inds, ner_inds, \
                bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = data
                outputs = self.bert_model(bert_batch_ids, token_type_ids=bert_seq_ids)
                # bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
                # loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
                loss, out_logits = self.get_ner(outputs[-1], bert2toks, ner_inds)
                # print("Predictions")
                # print(torch.argmax(out_logits,dim=2))
                # print("Trues")
                # print(ner_inds)
                if i % 100 == 99:
                    print("Average loss on {} batches : {}".format(i + 1, ner_loss / (i + 1)))
                loss.backward()
                self.ner_head.optimizer.step()
                self.bert_optimizer.step()
                ner_loss = ner_loss + loss.item()
            avg_ner_loss = ner_loss / eval_interval

            print("Average ner loss : {}".format(avg_ner_loss))
            avg_ner_losses.append(avg_ner_loss)
            print("Evaluation for epoch {} ".format(j))
            self.bert_model.eval()
            self.ner_head.eval()
            f1, p, r = self.eval_ner()
            # f1, p, r = 0, 0, 0
            print("F1 {}".format(f1))
            logging.info("F1 {}".format(f1))
            results.append([f1, p, r])
            if f1 > best_f1:
                best_epoch = j
                best_f1 = f1
                self.save_all_model(model_save_name)
        print("Average losses")
        print(avg_ner_losses)
        result_save_path = os.path.join(args.output_dir, args.ner_result_file)
        self.write_ner_result(result_save_path, ner_type, results, best_epoch)

    def write_ner_result(self, result_save_path, ner_type, results, best_epoch):
        logging.info("Writing  results for ner to {}".format(result_save_path))
        if not os.path.exists(result_save_path):
            s = "DATASET\tPRE\tREC\tF-1\n"
        else:
            s = ""
        with open(result_save_path, "a") as o:
            f1, p, r = results[best_epoch]
            s += "{}\t{}\t{}\t{}\n".format(ner_type, p, r, f1)
            o.write(s)

    ## Now we are using the same dataset for training and testing##
    def eval_ner(self):
        print("Starting evaluation for ner")
        self.ner_eval_reader.for_eval = True  ## This is necessary for not applying random sampling during evaluation!!!
        dataset = self.ner_eval_reader
        all_sents = []
        all_lens = []
        all_preds = []
        all_truths = []
        for i, batch in enumerate(dataset):
            tokens, bert_batch_after_padding, data = batch
            data = [d.to(self.device) for d in data]
            sent_lens, masks, tok_inds, ner_inds, \
            bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = data
            try:
                outputs = self.bert_model(bert_batch_ids, token_type_ids=bert_seq_ids)
            except:
                print("Problematic batch")
                print(bert_batch_ids)
                logging.info("Tokens")
                logging.info(tokens)
                print(tokens)
                continue
            # bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
            # loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
            preds, ner_inds = self.get_ner(outputs[-1], bert2toks, ner_inds, predict=True)
            tokens_ = tokens[-1]
            l = len(tokens_)
            # logging.info("NER INDS SHAPE {} ".format(ner_inds.shape))
            # logging.info("Predictions {} \n Truth {} ".format(preds[:l],ner_inds[:l]))
            all_sents.extend(tokens)
            all_lens.extend(sent_lens)
            all_preds.extend(preds)
            all_truths.extend(ner_inds)
            # if i % 50 == 49:
            #     logging.info("Processed {} batches".format(i + 1))
            # break

        # print(all_sents)
        # print(all_truths)
        # print(all_preds)
        # print(all_lens)
        sents = generate_pred_content(all_sents, all_preds, all_truths, all_lens, self.args.ner_label_vocab)
        orig_idx = dataset.orig_idx
        sents = unsort_dataset(sents, orig_idx)
        conll_file = os.path.join(self.args.output_dir,'ner_out')
        transition_file = os.path.join(self.args.output_dir,'crf_transitions')

        with open(transition_file,"a") as o:
            transitions = to_list(self.ner_head.classifier.transition)
            print("Transitions")
            print(transitions)
            logging.info("Transitions")
            logging.info(transitions)
            o.write("\n".join([" ".join([str(trans) for trans in transition]) for transition in transitions]))
            o.write("\n")
        conll_writer(conll_file, sents, ["token", 'truth', "ner_pred"], "ner")
        # prec, rec, f1 = 0,0,0
        prec, rec, f1 = evaluate_conll_file(open(conll_file, encoding='utf-8').readlines())
        print("NER Precision : {}  Recall : {}  F-1 : {}".format(prec, rec, f1))
        logging.info("NER Precision : {}  Recall : {}  F-1 : {}".format(prec, rec, f1))
        return round(f1, 2), round(prec, 2), round(rec, 2)


def main():
    biomlt = BioMLT()
    qa_types = ['yesno', 'list', 'factoid']
    mode = biomlt.args.mode
    predict = biomlt.args.predict
    if mode == "qas":
        if predict:
            biomlt.load_qas_data(biomlt.args, qa_types=qa_types, for_pred=True)
            biomlt.run_test()
            # biomlt.load_train_data()
        else:
            print("Running train_qas")
            biomlt.train_qas()
    elif mode == "joint_flat":
        if predict:
            biomlt.load_qas_data(biomlt.args, qa_types=qa_types, for_pred=True)
            biomlt.run_test()
        else:
            biomlt.train_qas_ner()
    elif mode == "ner":
        biomlt.train_ner()


if __name__ == "__main__":
    main()
