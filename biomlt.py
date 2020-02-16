from transformers import *

from transformers import get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
        #compute_predictions_logits,
            squad_evaluate,
            )
from squad_metrics import compute_predictions_logits
from conll_eval import evaluate_conll_file
import json
import copy
import torch
import torchvision
import random
import os
import torch.nn as nn
import torch.optim as optim
from reader import TrainingInstance, BertPretrainReader, MyTextDataset, mask_tokens, pubmed_files, squad_load_and_cache_examples
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
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
pretrained_bert_name  = 'bert-base-cased'
gettime = lambda x=datetime.datetime.now() : "{}_{}_{}_{}".format(x.month,x.day,x.hour,x.minute)


exp_prefix = gettime()
print("Time  {} ".format(exp_prefix))
random_seed = 12345
rng = random.Random(random_seed)
log_path = 'main_logger'
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def hugging_parse_args():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--load_model_path", default=None, type=str, required=False, help="The path to load the model to continue training."
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
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--nbest_path",
        default=None,
        type=str,
        required=False,
        help="The output path for storing nbest predictions. Used for evaluating with the bioasq scripts",
    )
    parser.add_argument(
        "--output_dir",
        default='save_dir',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str,default='bert', required=False, help="The model architecture to be trained or fine-tuned.",
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
        default='squad_data',
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--cache_folder",
        default="mlm_cache_folder",
        type=str,
        help="Directory to cache the mlm features",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--example_num",
        default=10,
        type = int,
        help = "Number of examples to train the data"
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
        "--squad_train_file",
        default="/home/aakdemir/biobert_data/BioASQ-6b/train/Snippet-as-is/BioASQ-train-factoid-6b-snippet-annotated.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--squad_yes_no",
        default=False,
        action = "store_true"
    )

    parser.add_argument(
        "--squad_predict_file",
        default="/home/aakdemir/biobert_data/BioASQ-6b/train/Snippet-as-is/BioASQ-train-factoid-6b-snippet-annotated.json",
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
        default=64,
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
        default="../biobert_data/biobert_v1.1_pubmed",
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
        default=False,
        action="store_true",
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--model_save_name", default= None, type=str, help="Model name to save"
    )
    parser.add_argument(
        "--mode", default= "qas", choices = ['qas','joint_flat','ner','qas_ner'], help="Determine which mode to use the Multi-tasking framework"
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
    parser.add_argument("--predict", default = False, action="store_true", help="Whether to run prediction only")
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
        "--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform."
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
    
    

    parser.add_argument('--ner_train_file', type=str, default='bc2gm_train.tsv', help='training file for ner')
    parser.add_argument('--ner_dev_file', type=str, default='bc2gm_train.tsv', help='training file for ner')
    parser.add_argument('--ner_test_file', type=str, default='bc2gm_train.tsv', help='training file for ner')

    #parser.add_argument('--output_dir', type=str, default='save_dir', help='Directory to store models and outputs')
    #parser.add_argument("--load_model", default=False, action="store_true", help="Whether to load a model or not")
    parser.add_argument('--ner_lr', type=float, default=0.0015, help='Learning rate for ner lstm')
    parser.add_argument("--qas_lr", default=5e-6, type=float, help="The initial learning rate for Qas.")
    parser.add_argument("--qas_adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    
    parser.add_argument('--qas_out_dim', type=int, default=2, help='Output dimension for question answering head')

    #parser.add_argument("--warmup_steps", default=5, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--t_total", default=5000, type=int, help="Total number of training steps")
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--ner_batch_size',
                        type=int, default=12, help='NER Batch size token based (not sentence)')
    parser.add_argument('--eval_batch_size', type=int, default=12, help='Batch size')
    #parser.add_argument('--block_size', type=int, default=128, help='Block size')
    #parser.add_argument('--epoch_num', type=int, default=20, help='Number of epochs')


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
            l = list(zip(sent[s_ind:sent_len-e_ind], truth[s_ind:sent_len-e_ind], pred[s_ind:sent_len-e_ind]))
            sents.append(l)
    else:
        for sent,pred in zip(tokens,preds):
            end_ind = -1
            s_ind = 1
            sents.append(list(zip(sent[s_ind:end_ind],pred[s_ind:end_ind])))
    
    return sents

# Wrapper class to combine NER with bioasq
class BioMTL(nn.Module):
    def __init__(self):
        super(BioMLT,self).__init__()
        self.BioMLT = BioMLT()
        self.args = self.BioMLT.args
        self.ner_head = NerModel(self.args)

class BioMLT(nn.Module):
    def __init__(self):
        super(BioMLT,self).__init__()
        self.args = hugging_parse_args()
        self.device = self.args.device
        #try:
        if self.args.biobert_model_path is not None and not self.args.init_bert:
            print("Trying to load from {} ".format(self.args.biobert_model_path))
            self.bert_model = BertForPreTraining.from_pretrained(self.args.biobert_model_path,
            from_tf=True,output_hidden_states=True)
        #except:
            #logging.info("Could not load biobert model loading from {}  ".format(pretrained_bert_name))
            #print("Could not load biobert model loading from {}  ".format(pretrained_bert_name))
        else:
            pretrained_bert_name = self.args.model_name_or_path
            if pretrained_bert_name is None:
                print("BERT model name should not be empty when init_model is given")
            if self.args.mlm :
                self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name,output_hidden_states=True)
            else:
                self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name, output_hidden_states=True)

        #print(self.bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert_out_dim = self.bert_model.bert.encoder.layer[11].output.dense.out_features
        self.args.bert_output_dim = self.bert_out_dim
        print("BERT output dim {}".format(self.bert_out_dim))

        self.ner_path = self.args.ner_train_file

        #self.ner_reader = DataReader(self.ner_path, "NER",tokenizer=self.bert_tokenizer,batch_size = 30)
        #self.args.ner_label_vocab = self.ner_reader.label_voc
        #self.ner_head = NerModel(self.args)

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
        self.yesno_optimizer = optim.AdamW([{"params": self.yesno_head.parameters()}],
                                      lr=self.yesno_lr, eps=self.args.qas_adam_epsilon)


        self.qas_head  = QasModel(self.args)

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
    def load_all_model(self,load_path): 
        #self.jointmodel=JointModel(self.args)
        load_path = self.args.load_model_path
        #save_path = os.path.join(self.args['save_dir'],self.args['save_name'])
        logging.info("Model loaded  from: %s"%load_path)
        loaded_params = torch.load(load_path)
        print("My params before loading")
        my_dict = self.state_dict()
        print("Yes-no head weights before loading")
        before = self.yesno_head.weight[:10]
        print(before)
        pretrained_dict = {k: v for k, v in loaded_params.items() if k in self.state_dict()}
        my_dict.update(pretrained_dict)
        self.load_state_dict(my_dict)
        print("My params after  loading")
        print("Yes-no head weights after loading")
        print(self.yesno_head.weight[:3])

    def save_all_model(self,save_path=None,weights=True):
        if self.args.model_save_name is None and save_path is None:
            save_name = os.path.join(self.args.output_dir,"{}_{}".format(self.args.mode,exp_prefix))
        else:
            if save_path is None:
                save_path = self.args.model_save_name
            save_name = os.path.join(self.args.output_dir,save_path)
        if weights:
            logging.info("Saving biomlt model to {}".format(save_name))
            torch.save(self.state_dict(), save_name)
        config_path = os.path.join(self.args.output_dir,self.args.config_file)
        arg = copy.deepcopy(self.args)
        del arg.device
        if hasattr(arg,"ner_label_vocab") :
            del arg.ner_label_vocab
        arg = vars(arg)
        with open(config_path,'w') as outfile:
            json.dump(arg,outfile)

    ## We are now averaging over the bert layer outputs for the NER task
    ## We may want to do this for QAS as well?
    ## This is very slow, right?
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


    def _get_token_to_bert_predictions(self,predictions, bert2toks):
        #logging.info("Predictions shape {}".format(predictions.shape))

        #logging.info("Bert2toks shape {}".format(bert2toks.shape))
        bert_predictions = []
        for pred,b2t in zip(predictions,bert2toks):
            bert_preds = []
            for b in b2t:
                bert_preds.append(pred[b])
            stack = torch.stack(bert_preds)
            bert_predictions.append(stack)
        stackk = torch.stack(bert_predictions)
        return stackk


    def _get_squad_bert_batch_hidden(self,hiddens, layers = [-2,-3,-4]):
        return torch.mean(torch.stack([hiddens[i] for i in layers]),0)

    def _get_squad_to_ner_bert_batch_hidden(self, hiddens , bert2toks, layers=[-2,-3,-4],device='cpu'):
        pad_size = hiddens[-1].shape[1]
        hidden_dim = hiddens[-1].shape[2]
        pad_vector = torch.tensor([0.0 for i in range(hidden_dim)]).to(device)
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]),0)
        batch_my_hiddens = []
        batch_lens = []
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
            batch_lens.append(len(my_hiddens))
            for i in range(pad_size - len(my_hiddens)):
                my_hiddens.append(pad_vector)
            sent_hiddens = torch.stack(my_hiddens)
            batch_my_hiddens.append(sent_hiddens)
        #for sent_hidden in batch_my_hiddens:
            #logging.info("Squad squeezed sent shape {}".format(sent_hidden.shape))
        return torch.stack(batch_my_hiddens),torch.tensor(batch_lens)

    def load_model(self):
        if self.args.mlm:
            logging.info("Attempting to load  model from {}".format(self.args.output_dir))
            self.bert_model = BertForMaskedLM.from_pretrained(self.args.output_dir)
        else:
            self.bert_model = BertForPreTraining.from_pretrained(self.args.output_dir)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.output_dir)
        sch_path = os.path.join(self.args.output_dir,"scheduler.pt")
        opt_path =os.path.join(self.args.outpt_dir,"optimizer.pt")
        if os.path.isfile(sch_path) and os.path.isfile(opt_path):
            self.bert_optimizer.load_state_dict(torch.load(opt_path))
            self.bert_scheduler.load_state_dict(torch.load(sch_path))
        logging.info("Could not load model from {}".format(self.args.output_dir))
        logging.info("Initializing Masked LM from {} ".format(pretrained_bert_name))
        #self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name)
        #self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name)
    
    def forward(self):
        return 0 
    

    def save_model(self):
        out_dir =self.args.output_dir
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

    def evaluate_qas(self,ind,only_preds = False):
        device = self.args.device
        self.device = device
        args =self.args
        if self.args.model_save_name is None:
            prefix = gettime()+"_"+str(ind)
        else : 
            prefix = self.args.model_save_name
        qas_eval_dataset,examples,features = squad_load_and_cache_examples(args,self.bert_tokenizer,evaluate=True,output_examples=True)
        print("Size of the test dataset {}".format(len(qas_eval_dataset)))
        eval_sampler = SequentialSampler(qas_eval_dataset)
        eval_dataloader = DataLoader(qas_eval_dataset, sampler=eval_sampler,batch_size = args.eval_batch_size)
        logger.info("Evaluation {} started".format(ind))
        logger.info("***** Running evaluation {} with only_preds = {}*****".format(prefix,only_preds))
        logger.info("  Num examples = %d", len(qas_eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_results = []
        for batch in tqdm(eval_dataloader, desc = "Evaluating"):
            self.bert_model.eval()
            self.qas_head.eval()
            batch = tuple(t.to(self.device) for t in batch)
            #print("Batch shape  {}".format(batch[0].shape))
            #if len(batch[0].shape)==1:
            #    batch = tuple(t.unsqueeze_(0) for t in batch)
            #logging.info(batch[0])
            squad_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                #"start_positions": batch[3],
                #"end_positions": batch[4],
            }
            bert_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            with torch.no_grad():
                outputs = self.bert_model(**bert_inputs)
                #squad_inputs["bert_outputs"] = outputs[-1][-2]
                
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                #logging.info("Bert out shape {}".format(bert_out.shape))
                qas_out = self.get_qas(bert_out,
                                       batch,
                                       eval=True,
                                       is_yes_no=self.args.squad_yes_no)
                #qas_out = self.qas_head(**squad_inputs)
                #print(qas_out)
                #loss,  start_logits, end_logits = qas_out
                #length = torch.sum(batch[1])
                #start_logits = start_logits.cpu().detach().numpy()
                #end_logits = end_logits.cpu().detach().numpy()
                #tokens = self.bert_tokenizer.convert_ids_to_tokens(batch[0].squeeze(0).detach().cpu().numpy()[:length])
                example_indices = batch[3]
                #print("Example indices inside evaluate_qas {}".format(example_indices))
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                if self.args.squad_yes_no:
                    output = qas_out[i,:].detach().cpu().numpy()
                    yesno_logit = output
                    print("What is start_logit {}".format(yesno_logit))
                    probs = self.yesno_soft(torch.tensor(yesno_logit).unsqueeze(0))
                    print("Yes-no probs : {}".format(probs))
                    result = SquadResult(unique_id,
                                         float(yesno_logit[0]),float(yesno_logit[1]))
                else:
                    output = [to_list(output[i]) for output in qas_out]
                    start_logit,end_logit = output
                    result = SquadResult(unique_id , start_logit, end_logit)
                #print(result.start_logits)
                all_results.append(result) 

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
        if args.nbest_path is not None:
            output_nbest_file = args.nbest_path
        else:
            output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix)) 

        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
        print("Length of predictions {} feats  {} examples {} ".format(len(all_results),len(examples),len(features)))
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
            is_yes_no=self.args.squad_yes_no
        )   

        if only_preds : 
            return output_nbest_file, output_prediction_file
        print("example answer:: ")
        print(examples[0].answers)
        results = squad_evaluate(examples, predictions)
        f1 = results['f1']
        exact = results['exact']
        total = results['total']
        print("RESULTS : f1 {}  exact {} total {} ".format(f1, exact, total))
        logging.info("RESULTS : f1 {} exact {} total {} ".format(f1, exact, total))
        return f1, exact, total


    def predict_qas(self,batch):
        ## batch_size = 1
        if len(batch[0].shape)==1:
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
            start_pred,end_pred = self.qas_head.predict(**squad_inputs)
            length = torch.sum(batch[1])
            tokens = self.bert_tokenizer.convert_ids_to_tokens(batch[0].squeeze(0).detach().cpu().numpy()[:length])
        logging.info("Example {}".format(tokens))
        logging.info("Answer {}".format(tokens[start_pred:end_pred+1]))
        logging.info("Start Pred {}  start truth {}".format(start_pred,squad_inputs["start_positions"]))
        logging.info("End Pred {}  end truth {}".format(end_pred,squad_inputs["end_positions"]))
     
    def get_qas(self, bert_output, batch, eval=False,is_yes_no = False):

        #batch = tuple(t.unsqueeze_(0) for t in batch)
        if eval:
            
            squad_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                'bert_outputs' : bert_output
                #"start_positions": batch[3],
                #"end_positions": batch[4],
            }
        else:

            squad_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "bert_outputs" : bert_output
            }

        if not is_yes_no:
            qas_outputs = self.qas_head(**squad_inputs)
        else:
            ##!!!  CLS TOKEN  !!! ##
            yes_no_logits = self.yesno_head(bert_output[:,0])
            if not eval:
                loss = self.yesno_loss(yes_no_logits, batch[3])
                return (loss, yes_no_logits)
            return yes_no_logits
        #print(qas_outputs[0].item())
        #qas_outputs[0].backward()
        #self.bert_optimizer.step()
        #self.qas_head.optimizer.step()
        return qas_outputs

    def get_ner(self,bert_output,bert2toks,ner_inds=None,predict=False):
        bert_hiddens = self._get_bert_batch_hidden(bert_output,bert2toks)

        if predict:
            all_preds = []
            out_logits = self.ner_head(bert_hiddens,ner_inds,pred=predict)
            voc_size = len(self.ner_reader.label_vocab)
            #print(bert2toks[-1])
            preds = torch.argmax(out_logits,dim=2).detach().cpu().numpy()//voc_size
            #print("Preds ", preds)
            for pred in preds:
                ## MAP [CLS] and [SEP] predictions to O
                p = list(map(lambda x : "O" if (x == "[SEP]" or x == "[CLS]" or x == "[PAD]") else x,
                              self.ner_reader.label_vocab.unmap(pred)))
                all_preds.append(p)
            all_ner_inds = []
            if ner_inds is not None :
                ner_inds = ner_inds.detach().cpu().numpy()//voc_size
                for n in ner_inds:
                    n_n = list(map(lambda x: "O" if (x == "[SEP]" or x == "[CLS]" or x == "[PAD]") else x,
                                 self.ner_reader.label_vocab.unmap(n)))
                    all_ner_inds.append(n_n)
                return all_preds, all_ner_inds
            else:
                return all_preds
        #logging.info("NER output {} ".format(ner_outs.))
        else:
            ner_outs = self.ner_head(bert_hiddens,ner_inds)
            return ner_outs

    def run_test(self):
        assert self.args.load_model_path is not None, "Model path to be loaded must be defined to run in predict mode!!!"
        self.load_all_model(self.args.load_model_path)

        device = self.args.device
        self.bert_model.to(device)
        self.qas_head.to(device)
        nbest_file, pred_file = self.evaluate_qas(0,only_preds=True)
        print("Predictions are saved to {} \n N-best predictions are saved to {} ".format(pred_file,nbest_file))       



    ## training a flat model (multi-task learning hard-sharing)
    def train_qas_ner(self):

        # now initializing Ner head here !!
        print("Reading NER data from {}".format(self.ner_path))
        self.ner_reader = DataReader(
            self.ner_path, "NER", tokenizer=self.bert_tokenizer,
            batch_size=self.args.ner_batch_size)
        self.args.ner_label_vocab = self.ner_reader.label_vocab
        print(self.args.ner_label_vocab.w2ind)
        self.ner_head = NerModel(self.args)
        device = self.args.device
        self.device = device
        args =hugging_parse_args()
        self.huggins_args = args
        qas_train_dataset = squad_load_and_cache_examples(args,
                                                          self.bert_tokenizer,
                                                          yes_no=self.args.squad_yes_no)
        print("Size of the dataset {}".format(len(qas_train_dataset)))
        args.train_batch_size = self.args.batch_size
        qas_train_sampler = RandomSampler(qas_train_dataset)
        qas_train_dataloader = DataLoader(qas_train_dataset,
                                          sampler=qas_train_sampler,
                                          batch_size=args.train_batch_size)
        t_totals = len(qas_train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        #self.train_ner()
        epochs_trained = 0

        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.ner_head.to(device)
        self.bert_model.train()
        self.qas_head.train()
        self.ner_head.train()
        for index, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(qas_train_dataloader, desc="Iteration")
            self.bert_model.train()
            self.qas_head.train()
            self.ner_head.train()
            for step, batch in enumerate(epoch_iterator):

                # empty gradients
                self.bert_optimizer.zero_grad()
                self.qas_head.optimizer.zero_grad()
                self.ner_head.optimizer.zero_grad()

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
                #logging.info("Number of sentences in the ner batch : {}".format(len(tokens)))
                #logging.info("Number of sentences in the dep batch : {}".format(batch[0].shape[0]))

                # ner output
                outputs = self.bert_model(bert_batch_ids, token_type_ids=bert_seq_ids)
                # bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
                # loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
                ner_loss, ner_out_logits = self.get_ner(outputs[-1], bert2toks, ner_inds)
                logging.info("NER out shape : {}".format(ner_out_logits.shape))


                # for hierarchical setting
                # bert_outs_for_ner, lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1], batch[-1],device=device)
                # ner_outs = self.ner_head(bert_outs_for_ner)
                # ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs, batch[-1])

                # qas output
                outputs = self.bert_model(**bert_inputs)
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                #logging.info("Bert out shape {}".format(bert_out.shape))
                qas_outputs = self.get_qas(bert_out, batch)
                #logging.info("QAS out shape : {}".format(qas_outputs[1].shape))
                #qas_outputs = self.qas_head(**squad_inputs)
                # print(qas_outputs[0].item())

                # sum losses for backprop
                total_loss = ner_loss + qas_outputs[0]
                total_loss = ner_loss
                total_loss.backward()
                #logging.info("Total loss {}".format(total_loss.item()))
                logging.info("Total loss {} ner: {}  dep : {}".format(total_loss.item(),
                                                                       ner_loss.item(),qas_outputs[0].item()))

                # backpropagation
                self.ner_head.optimizer.step()
                self.bert_optimizer.step()
                # not sure if optimizer and scheduler works simultaneously
                #self.bert_scheduler.step()
                self.qas_head.optimizer.step()
            self.bert_model.eval()
            self.qas_head.eval()
            self.ner_head.eval()
            with torch.no_grad():
                self.evaluate_qas(index)
                self.eval_ner()
            #self.predict_ner()


    def predict_ner(self):
        self.eval_ner()


    def train_qas(self):
        device = self.args.device
        args =hugging_parse_args()
        print("Is yes no ? {}".format(self.args.squad_yes_no))
        train_dataset = squad_load_and_cache_examples(args,
                                                      self.bert_tokenizer,
                                                      yes_no =self.args.squad_yes_no)

        print("Training a model for {} type questions".
              format("YES-NO " if self.args.squad_yes_no else "FACTOID"))
        print("Size of the train dataset {}".format(len(train_dataset)))
        args.train_batch_size = self.args.batch_size
        #train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        t_totals = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": self.qas_head.parameters(), "weight_decay": 0.0}]
        #self.bert_squad_optimizer =AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        ## Scheduler for sub-components
        #scheduler = get_linear_schedule_with_warmup(
        #self.bert_squad_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_totals)
        tr_loss, logging_loss = 0.0, 0.0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        best_result = 0
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.yesno_head.to(device)
        #self.ner_head.to(device)
        print("weights before training !!")
        print(self.qas_head.qa_outputs.weight[-10:])
        for epoch, _ in enumerate(train_iterator):
            total_loss = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                #print("BATCH")
                #print(batch)
                #if step >10:
                #    break
                self.bert_optimizer.zero_grad()
                self.qas_head.optimizer.zero_grad()
                self.yesno_optimizer.zero_grad()
                #batch = train_dataset[0]
                #batch = tuple(t.unsqueeze(0) for t in batch)
                #logging.info(batch[-1])
                self.bert_model.train()
                self.qas_head.train()
                #logging.info(self.bert_tokenizer.convert_ids_to_tokens(batch[0][0].detach().numpy()))
                batch = tuple(t.to(device) for t in batch)
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                #logging.info("Input ids shape : {}".format(batch[0].shape))
                #bert2toks = batch[-1]
                outputs = self.bert_model(**bert_inputs)
                #bert_outs_for_ner , lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1],batch[-1],device=device)
                #print("BERT OUTS FOR NER {}".format(bert_outs_for_ner.shape))
                #ner_outs = self.ner_head(bert_outs_for_ner)
                #ner_outs_2= self.get_ner(outputs[-1], bert2toks) 
                #ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs,batch[-1])
                #logging.info("NER OUTS FOR QAS {}".format(ner_outs_for_qas.shape))
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                #logging.info("Bert out shape {}".format(bert_out.shape))
                qas_outputs = self.get_qas(bert_out,batch,eval=False,is_yes_no=self.args.squad_yes_no)
                
                #qas_outputs = self.qas_head(**squad_inputs)
                #print(qas_outputs[0].item())
                loss = qas_outputs[0]
                loss.backward()
                self.bert_optimizer.step()
                self.qas_head.optimizer.step()
                self.yesno_optimizer.step()
                total_loss += loss.item()
                
                if step%500==499:
                    if self.args.model_save_name is None:
                        checkpoint_name = self.args.mode+"_"+exp_prefix+"_check_{}_{}".format(epoch,step)
                    else :
                        checkpoint_name = self.args.model_save_name+"_check_"+str(step)
                    logging.info("Saving checkpoint to {}".format(checkpoint_name))
                    self.save_all_model(checkpoint_name)
                    logging.info("Average loss after {} steps : {}".format(step+1,total_loss/(step+1)))
            print("Epoch {} is finished, moving to evaluation ".format(epoch))
            f1, exact, total  = self.evaluate_qas(epoch)
            if f1 >= best_result :
                best_result = f1
                print("Best f1 of {}".format(f1))
                save_name = "{}_{}".format(self.args.mode,exp_prefix) if self.args.model_save_name is None else self.args.model_save_name
                print("Saving best model with {}  to {}".format(best_result,
                    save_name))
                logging.info("Saving best model with {}  to {}".format(best_result,
                    save_name))
                self.save_all_model(save_name)


    def pretrain_mlm(self):
        device = self.args.device
        epochs_trained = 0
        epoch_num = self.args.epoch_num
        batch_size = self.args.batch_size
        block_size = self.args.block_size
        huggins_args =hugging_parse_args()
        
        #file_list = pubmed_files()
        file_list = ["PMC6958785.txt","PMC6961255.txt"]
        train_dataset = MyTextDataset(self.bert_tokenizer,huggins_args,file_list,block_size = block_size)
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
        #self.dataset = reader.create_training_instances(file_list,bert_tokenizer)
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
        #epochs_trained, int(huggins_args.num_train_epochs), desc="Epoch")
        epochs_trained, int(epoch_num), desc="Epoch")
    #set_seed(args)  # Added here for reproducibility
        for _ in train_iterator:
            for step, batch in enumerate(epoch_iterator):
                #print("Batch shape {} ".format(batch.shape))
                #print("First input {} ".format(batch[0]))
                self.bert_optimizer.zero_grad()            ## update mask_tokens to apply curriculum learnning!!!!
                inputs, labels = mask_tokens(batch, self.bert_tokenizer, huggins_args)
                tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs.cpu().detach().numpy()[0,:])
                label_tokens = self.bert_tokenizer.convert_ids_to_tokens(labels.cpu().detach().numpy()[0,:])
                logging.info("Tokens {}".format(tokens))
                logging.info("Labels ".format(label_tokens))
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.bert_model(inputs,masked_lm_labels=labels)
                loss = outputs[0]
                logging.info("Loss obtained for batch of {} is {} ".format(batch.shape,loss.item()))
                loss.backward()
                self.bert_optimizer.step()
                if step ==2:
                    break
            self.save_model()
            logging.info("Training is finished moving to evaluation")
            self.mlm_evaluate()

    def mlm_evaluate(self,prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = out_dir =self.args.output_dir

        #eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        eval_batch_size = 1
        file_list = ["PMC6958785.txt"]
        eval_dataset = MyTextDataset(self.bert_tokenizer,self.huggins_args,file_list,block_size = 128)
        #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        def collate(examples):
            return pad_sequence(examples, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=collate
        )

        # multi-gpu evaluate
        #if args.n_gpu > 1:
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
        reader = BertPretrainReader(file_list,self.bert_tokenizer)
        #dataset = reader.create_training_instances(file_list,self.bert_tokenizer)
        tokens = reader.dataset[1].tokens
        logging.info(tokens)
        input_ids = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0) # Batch size 1
        #token_type_ids= torch.tensor(dataset[1].segment_ids).unsqueeze(0)
        #print(input_ids.shape)
        #print(dataset[1].segment_ids)
        #next_label = torch.tensor([ 0 if dataset[1].is_random_next  else  1])
        token_ids, mask_labels, next_label, token_type_ids = reader[0]
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        for i in range(10):
            self.bert_optimizer.zero_grad()
            #print("Input shape {}".format(token_ids.shape))
            outputs = self.bert_model(token_ids,token_type_ids= token_type_ids)
            prediction_scores, seq_relationship_scores = outputs[:2]
            vocab_dim = prediction_scores.shape[-1]
            masked_lm_loss = loss_fct(prediction_scores.view(-1, vocab_dim), mask_labels.view(-1))
            next_sent_loss = loss_fct(seq_relationship_scores.view(-1,2),next_label.view(-1))
            loss = masked_lm_loss + next_sent_loss
            loss.backward()
            self.bert_optimizer.step()
        pred_tokens = self.bert_tokenizer.convert_ids_to_tokens(torch.argmax(prediction_scores,dim=2).detach().cpu().numpy()[0])
        logging.info("{} {} ".format("Real tokens", tokens))
        logging.info("{} {} ".format("Predictions", pred_tokens))


    def train_ner(self):
        self.ner_reader = DataReader(
            self.ner_path, "NER",tokenizer=self.bert_tokenizer,
            batch_size = self.args.ner_batch_size)

        self.args.ner_label_vocab = self.ner_reader.label_vocab
        self.ner_head = NerModel(self.args)
        device = self.device
        print("Starting training for NER ")
        print("Tokens  : ",self.ner_reader[0][0])
        self.bert_model.to(device)
        self.ner_head.to(device)
        self.bert_model.train()
        self.ner_head.train()
        for j in range(10):
            for i in range(10):
                self.bert_optimizer.zero_grad()
                self.ner_head.optimizer.zero_grad()
                tokens, bert_batch_after_padding, data = self.ner_reader[i]
                print("Number of sentences in the batch : {}".format(len(tokens)))
                data = [d.to(device) for d in data]
                sent_lens, masks, tok_inds, ner_inds,\
                     bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data
                outputs = self.bert_model(bert_batch_ids, token_type_ids = bert_seq_ids)
                #bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
                #loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
                loss, out_logits = self.get_ner(outputs[-1],bert2toks,ner_inds)
                print(loss.item())
                loss.backward()
                self.ner_head.optimizer.step()
                self.bert_optimizer.step()
            self.eval_ner()

    def eval_ner(self):
        print("Starting evaluation for ner")
        self.ner_reader.for_eval = True
        dataset  = self.ner_reader
        all_sents = []
        all_lens = []
        all_preds = []
        all_truths = []
        for i, batch in enumerate(dataset):
            tokens, bert_batch_after_padding, data = batch
            data = [d.to(self.device) for d  in data]
            sent_lens, masks, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data
            outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
            #bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
            #loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
            preds, ner_inds = self.get_ner(outputs[-1],bert2toks,ner_inds,predict = True)
            tokens_= tokens[-1]
            l = len(tokens_)
            #logging.info("NER INDS SHAPE {} ".format(ner_inds.shape))
            #logging.info("Predictions {} \n Truth {} ".format(preds[:l],ner_inds[:l]))
            all_sents.extend(tokens)
            all_lens.extend(sent_lens)
            all_preds.extend(preds)
            all_truths.extend(ner_inds)
        #print(all_sents)
        #print(all_truths)
        #print(all_preds)
        #print(all_lens)
        sents = generate_pred_content(all_sents,all_preds,all_truths, all_lens,self.args.ner_label_vocab)
        orig_idx = dataset.orig_idx
        logging.info("Original indexes")
        logging.info(orig_idx)
        logging.info("Unsorting the ner data")
        logging.info(sents)
        sents = unsort_dataset(sents, orig_idx)
        logging.info("after sorting")
        logging.info(sents)
        conll_file = 'ner_out'
        conll_writer(conll_file, sents, ["token", 'truth',"ner_pred"],"ner")
        prec, rec, f1 = evaluate_conll_file(open(conll_file,encoding='utf-8').readlines())
        print("Precision : {}  Recall : {}  F-1 : {}".format(prec,rec,f1))
def main():
    biomlt = BioMLT()
    mode = biomlt.args.mode
    predict = biomlt.args.predict
    if mode == "qas" :
        if predict:
            biomlt.run_test()
        else:
            print("Running train_qas")
            biomlt.train_qas()
    elif mode == "joint_flat":
        biomlt.train_qas_ner()
    elif mode == "ner":
        biomlt.train_ner()
    #biomlt.train_qas_ner()
    #biomlt.pretrain_mlm()
    #mymodel = BertForMaskedLM.from_pretrained("save_dir",output_hidden_states=True)
    #config = BertConfig.from_json_file(args.biobert_tf_config)
    #biobert = BertModel.from_pretrained(args.biobert_tf_model,config=config,from_tf=True)
    #print(biobert)
if __name__=="__main__":
    main()
