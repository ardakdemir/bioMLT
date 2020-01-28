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
        default=384,
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
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--block_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs')
    parser.add_argument('--mlm', type=bool, default=True, help='To train a mlm only pretraining model')
    args = vars(parser.parse_args())
    args['device'] = device
    return args
class BioMLT():
    def __init__(self):
        self.args = parse_args()
        self.qa_output_labels = 2
        if self.args['mlm'] :
            self.bert_model = BertForMaskedLM.from_pretrained(pretrained_bert_name,output_hidden_states=True)
        else:
            self.bert_model = BertForPreTraining.from_pretrained(pretrained_bert_name, output_hidden_states=True)
        #print(self.bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.ner_path = self.args['ner_train_file']
        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]
        self.bert_out_dim = self.bert_model.bert.encoder.layer[11].output.dense.out_features
        print("BERT output dim {}".format(self.bert_out_dim))

        self.qas_head  = nn.Linear(self.bert_out_dim, self.qa_output_labels)
        self.args['bert_output_dim'] = self.bert_out_dim
        self.bert_optimizer = AdamW(optimizer_grouped_parameters,
                         lr=2e-5)

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



    def train_squad(self):
        device = self.args['device']
        args =hugging_parse_args()
        train_dataset = squad_load_and_cache_examples(args,self.bert_tokenizer)
        args.train_batch_size = self.args['batch_size']
        #train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        t_totals = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": self.qas_head.parameters(), "weight_decay": 0.0}]
        self.bert_squad_optimizer =AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
        self.bert_squad_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_totals)
        tr_loss, logging_loss = 0.0, 0.0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        self.bert_optimizer.zero_grad()
        self.bert_squad_optimizer.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch")
        # Added here for reproductibility
        self.bert_model.to(device)
        self.qas_head.to(device)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.bert_model.train()
                batch = tuple(t.to(device) for t in batch)

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
                # model outputs are always tuple in transformers (see doc)
                print("Bert output shape {} ".format(outputs[-1][-2].shape))
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
        self.ner_reader = DataReader(self.ner_path, "NER",tokenizer=self.bert_tokenizer)
        self.args['ner_label_vocab'] = self.ner_reader.label_voc
        self.ner_head = NerModel(self.args)
        print("Starting training")
        for i in range(10):
            self.ner_head.optimizer.zero_grad()
            tokens, bert_batch_after_padding, data = self.ner_reader[0]
            sent_lens, masks, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data
            outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
            bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
            loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
            print(loss.item())
            loss.backward()
            self.ner_head.optimizer.step()
        self.eval_ner()
    def eval_ner(self):
        tokens, bert_batch_after_padding, data = self.ner_reader[0]
        sent_lens, masks, tok_inds, ner_inds,\
             bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = data

        outputs = self.bert_model(bert_batch_ids,token_type_ids= bert_seq_ids)
        bert_hiddens = self._get_bert_batch_hidden(outputs[-1],bert2toks)
        loss, out_logits =  self.ner_head(bert_hiddens,ner_inds)
        print("Predictions {}".format(out_logits.shape))
        print(torch.argmax(out_logits,dim=2))
        print("Labels")
        print(ner_inds)
if __name__=="__main__":

    biomlt = BioMLT()
    biomlt.train_squad()
    #biomlt.pretrain_mlm()
