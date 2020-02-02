from transformers import *
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
from nerreader import DataReader
from nermodel import NerModel
from qasmodel import QasModel
import argparse
from torch.nn import CrossEntropyLoss, MSELoss

pretrained_bert_name  = 'bert-base-cased'

random_seed = 12345
rng = random.Random(random_seed)
log_path = 'main_logger'
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')



def hugging_parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=False, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str,default='bert', required=False, help="The model architecture to be trained or fine-tuned.",
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
        default=100,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--example_num",
        default=10,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--squad_train_file",
        default="train-v2.0.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--squad_predict_file",
        default="dev-v2.0.json",
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
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
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
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
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
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
    args = parser.parse_args()
    return args

def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))
    parser = argparse.ArgumentParser()

    parser.add_argument('--ner_train_file', type=str, default='bc2gm_train.tsv', help='training file for ner')
    parser.add_argument('--output_dir', type=str, default='save_dir', help='training file for ner')

    parser.add_argument('--ner_lr', type=float, default=0.0015, help='Learning rate for ner lstm')
    parser.add_argument("--qas_lr", default=5e-5, type=float, help="The initial learning rate for Qas.")
    parser.add_argument("--qas_adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")


    parser.add_argument('--qas_out_dim', type=int, default=2, help='Output dimension for question answering head')


    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--block_size', type=int, default=32, help='Block size')
    parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs')


    parser.add_argument('--mlm', type=bool, default=True, help='To train a mlm only pretraining model')
    args = vars(parser.parse_args())
    args['device'] = device
    return args
class BioMLT():
    def __init__(self):
        self.args = parse_args()
        if self.args['mlm'] :
            self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name,output_hidden_states=True)
        else:
            self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name, output_hidden_states=True)
        #print(self.bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.bert_out_dim = self.bert_model.bert.encoder.layer[11].output.dense.out_features
        self.args['bert_output_dim'] = self.bert_out_dim
        print("BERT output dim {}".format(self.bert_out_dim))

        self.ner_path = self.args['ner_train_file']
        self.ner_reader = DataReader(self.ner_path, "NER",tokenizer=self.bert_tokenizer,batch_size = 30)
        self.args['ner_label_vocab'] = self.ner_reader.label_voc
        self.ner_head = NerModel(self.args)

        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]


        self.qas_head  = QasModel(self.args)

        self.bert_optimizer = AdamW(optimizer_grouped_parameters,
                         lr=2e-5)

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
        logging.info("Predictions shape {}".format(predictions.shape))

        logging.info("Bert2toks shape {}".format(bert2toks.shape))
        bert_predictions = []
        for pred,b2t in zip(predictions,bert2toks):
            bert_preds = []
            for b in b2t:
                bert_preds.append(pred[b])
            stack = torch.stack(bert_preds)
            #print(stack.shape)
            bert_predictions.append(stack)
        stackk = torch.stack(bert_predictions)
        print(stackk.shape)
        return stackk


    def _get_squad_bert_batch_hidden(self,hiddens, layers = [-2,-3,-4]):
        return torch.mean(torch.stack([hiddens[i] for i in layers]),0)

    def _get_squad_to_ner_bert_batch_hidden(self, hiddens , bert2toks, layers=[-2,-3,-4]):
        pad_size = hiddens[-1].shape[1]
        hidden_dim = hiddens[-1].shape[2]
        pad_vector = torch.tensor([0.0 for i in range(hidden_dim)])
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
    def save_model(self):
        out_dir =self.args['output_dir']
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print("Saving model checkpoint to %s", out_dir)
        logger.info("Saving model checkpoint to %s", out_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.bert_model
        model_to_save.save_pretrained(out_dir)
        self.bert_tokenizer.save_pretrained(out_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(out_dir, "training_args.bin"))


    def qas_predict(self,batch):
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
        outputs = self.bert_model(**bert_inputs)
        squad_inputs["bert_outputs"] = outputs[-1][-2]
        start_pred,end_pred = self.qas_head.predict(**squad_inputs)
        length = torch.sum(batch[1])
        tokens = self.bert_tokenizer.convert_ids_to_tokens(batch[0].squeeze(0).detach().numpy()[:length])
        logging.info("Example {}".format(tokens))
        logging.info("Answer {}".format(tokens[start_pred:end_pred+1]))
        logging.info("Start Pred {}  start truth {}".format(start_pred,squad_inputs["start_positions"]))
        logging.info("End Pred {}  end truth {}".format(end_pred,squad_inputs["end_positions"]))

    def get_qas(self, bert_output, batch):

        #batch = tuple(t.unsqueeze_(0) for t in batch)
        squad_inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        squad_inputs["bert_outputs"] = bert_output
        qas_outputs = self.qas_head(**squad_inputs)
        #print(qas_outputs[0].item())
        #qas_outputs[0].backward()
        #self.bert_optimizer.step()
        #self.qas_head.optimizer.step()
        return qas_outputs

    def get_ner(self,bert_output,bert2toks,ner_inds):
        bert_hiddens = self._get_bert_batch_hidden(bert_output,bert2toks)
        loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
        logging.info("NER loss {} ".format(loss.item()))
        return (loss,out_logits)



    ## training a flat model (multi-task learning hard-sharing)
    def train_qas_ner(self):
        device = self.args['device']
        args =hugging_parse_args()
        qas_train_dataset = squad_load_and_cache_examples(args,self.bert_tokenizer)
        print("Size of the dataset {}".format(len(qas_train_dataset)))
        args.train_batch_size = self.args['batch_size']
        qas_train_sampler = RandomSampler(qas_train_dataset)
        qas_train_dataloader = DataLoader(qas_train_dataset, sampler=qas_train_sampler, batch_size=args.train_batch_size)
        t_totals = len(qas_train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        #self.train_ner()
        epochs_trained = 0

        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        self.ner_head.to(device)

        for _ in train_iterator:
            epoch_iterator = tqdm(qas_train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.bert_optimizer.zero_grad()
                self.qas_head.optimizer.zero_grad()
                self.ner_head.optimizer.zero_grad()
                batch = qas_train_dataset[0]
                batch = tuple(t.unsqueeze(0) for t in batch)
                tokens, bert_batch_after_padding, data = self.ner_reader[0]
                #logging.info(batch[-1])
                self.bert_model.train()
                # logging.info(self.bert_tokenizer.convert_ids_to_tokens(batch[0][0].detach().numpy()))
                batch = tuple(t.to(device) for t in batch)
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                #logging.info("Input ids shape : {}".format(batch[0].shape))
                sent_lens, masks, tok_inds, ner_inds, \
                bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = data
                outputs = self.bert_model(bert_batch_ids, token_type_ids=bert_seq_ids)
                # bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
                # loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
                ner_loss, ner_out_logits = self.get_ner(outputs[-1], bert2toks, ner_inds)
                outputs = self.bert_model(**bert_inputs)
                bert_outs_for_ner, lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1], batch[-1])
                ner_outs = self.ner_head(bert_outs_for_ner)
                ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs, batch[-1])
                logging.info("BERT OUTS FOR NER {}".format(ner_outs_for_qas.shape))
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                logging.info("Bert out shape {}".format(bert_out.shape))
                qas_outputs = self.get_qas(bert_out, batch)
                # qas_outputs = self.qas_head(**squad_inputs)
                # print(qas_outputs[0].item())
                total_loss = ner_loss + qas_outputs[0]
                total_loss.backward()
                logging.info("TOtal loss {} {}  {}".format(total_loss.item(),
                                                           ner_loss.item(),qas_outputs[0].item()))
                self.ner_head.optimizer.step()
                self.bert_optimizer.step()
                self.qas_head.optimizer.step()

    def train_qas(self):
        device = self.args['device']
        args =hugging_parse_args()
        train_dataset = squad_load_and_cache_examples(args,self.bert_tokenizer)
        print("Size of the dataset {}".format(len(train_dataset)))
        args.train_batch_size = self.args['batch_size']
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

        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        for _ in train_iterator:
            epoch_iterator = tqdm(qas_train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.bert_optimizer.zero_grad()
                self.qas_head.optimizer.zero_grad()
                batch = train_dataset[0]
                batch = tuple(t.unsqueeze(0) for t in batch)
                logging.info(batch[-1])
                self.bert_model.train()
                #logging.info(self.bert_tokenizer.convert_ids_to_tokens(batch[0][0].detach().numpy()))
                batch = tuple(t.to(device) for t in batch)
                bert_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                logging.info("Input ids shape : {}".format(batch[0].shape))
                outputs = self.bert_model(**bert_inputs)
                bert_outs_for_ner , lens = self._get_squad_to_ner_bert_batch_hidden(outputs[-1],batch[-1])
                ner_outs = self.ner_head(bert_outs_for_ner)
                ner_outs_for_qas = self._get_token_to_bert_predictions(ner_outs,batch[-1])
                logging.info("BERT OUTS FOR NER {}".format(ner_outs_for_qas.shape))
                bert_out = self._get_squad_bert_batch_hidden(outputs[-1])
                logging.info("Bert out shape {}".format(bert_out.shape))
                qas_outputs = self.get_qas(bert_out,batch)
                #qas_outputs = self.qas_head(**squad_inputs)
                #print(qas_outputs[0].item())
                qas_outputs[0].backward()
                logging.info("Loss {}".format(qas_outputs[0].item()))
                self.bert_optimizer.step()
                self.qas_head.optimizer.step()
            self.qas_predict(batch)

    def pretrain_mlm(self):
        device = self.args['device']
        epochs_trained = 0
        epoch_num = self.args['epoch_num']
        batch_size = self.args['batch_size']
        block_size = self.args['block_size']
        huggins_args =hugging_parse_args()
        self.huggins_args = huggins_args
        file_list = pubmed_files()
        train_dataset = MyTextDataset(self.bert_tokenizer,huggins_args,file_list,block_size = block_size)
        print("Dataset size {} ".format(len(train_dataset)))
        train_sampler = RandomSampler(train_dataset)
        def collate(examples):
            return pad_sequence(examples, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate
        )
        #self.dataset = reader.create_training_instances(file_list,bert_tokenizer)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
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
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.bert_model(inputs,masked_lm_labels=labels)
                loss = outputs[0]
                logging.info("Loss obtained for batch of {} is {} ".format(batch.shape,loss.item()))
                loss.backward()
                self.bert_optimizer.step()
                if step ==2:
                    break
        #self.save_model()
        logging.info("Training is finished moving to evaluation")
        self.mlm_evaluate()

    def mlm_evaluate(self,prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = out_dir =self.args['output_dir']

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
            inputs = inputs.to(self.args['device'])
            labels = labels.to(self.args['device'])

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
        pred_tokens = self.bert_tokenizer.convert_ids_to_tokens(torch.argmax(prediction_scores,dim=2).detach().numpy()[0])
        logging.info("{} {} ".format("Real tokens", tokens))
        logging.info("{} {} ".format("Predictions", pred_tokens))


    def train_ner(self):
        self.ner_reader = DataReader(self.ner_path, "NER",tokenizer=self.bert_tokenizer,batch_size = 30)
        self.args['ner_label_vocab'] = self.ner_reader.label_voc
        self.ner_head = NerModel(self.args)
        print("Starting training")
        for j in range(10):
            for i in range(10):
                self.ner_head.optimizer.zero_grad()
                tokens, bert_batch_after_padding, data = self.ner_reader[0]
                sent_lens, masks, tok_inds, ner_inds,\
                     bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data
                outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
                #bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
                #loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
                loss, out_logits = self.get_ner(outputs[-1],bert2toks,ner_inds)
                #print(loss.item())
                loss.backward()
                self.ner_head.optimizer.step()
            self.eval_ner()
    def eval_ner(self):
        tokens, bert_batch_after_padding, data = self.ner_reader[0]
        sent_lens, masks, tok_inds, ner_inds,\
             bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data

        outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
        #bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
        #loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
        loss, out_logits = self.get_ner(outputs[-1],bert2toks,ner_inds)
        logging.info("Tokens")
        logging.info(tokens)
        voc_size = len(self.ner_reader.label_voc)
        preds = torch.argmax(out_logits,dim=2).detach().numpy()[0,:len(tokens)]//voc_size
        ner_inds = ner_inds.detach().numpy()[0, :len(tokens)]//voc_size
        logging.info("NER INDS {}".format(ner_inds))
        preds = self.ner_reader.label_voc.unmap(preds)
        ner_inds = self.ner_reader.label_voc.unmap(ner_inds)
        logging.info("Predictions {} \n Truth {} ".format(preds,ner_inds))
def main():
    biomlt = BioMLT()
    # biomlt.train_ner()
    biomlt.train_qas_ner()
    # biomlt.pretrain_mlm()

if __name__=="__main__":
    main()

