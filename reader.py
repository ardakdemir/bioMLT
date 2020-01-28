import tokenization
from transformers import *
from transformers import squad_convert_examples_to_features
import random
import collections
import logging
import torch
import pickle
import os
import argparse
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
random_seed = 12345
rng = random.Random(random_seed)
log_path = 'read_logger'
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')
pretrained_bert_name  = 'bert-base-uncased'


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

def pubmed_files(root = "/home/aakdemir/pubmed/pub/pmc/oa_bulk/"):
    if not os.path.isdir(root):
        return ["PMC6961255.txt","PMC6958785.txt"]
    #root = "/home/aakdemir/pubmed/pub/pmc/oa_bulk/"
    folder_list = os.listdir(root)
    file_list = []
    names = set()
    for folder in folder_list:
        fold_path = os.path.join(root,folder)
        if os.path.isdir(fold_path):
            for file in os.listdir(fold_path):
                if file.endswith(".txt"):
                    if file in names:
                        print("File with name {} exists in multiple folders")
                    names.add(file)
                    file_list.append(os.path.join(root,folder,file))
    print("Read {} files ".format(len(file_list)))
    return file_list


def squad_load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cache_folder = "squad_cache"
    #input_dir = args.squad_dir if args.data_dir else "."
    input_dir = args.squad_dir
    cached_features_file = os.path.join(cache_folder,
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)+".txt",
        ),
    )
    print("Cache path {} ".format(cached_features_file))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not input_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            #processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            processor = SquadV2Processor()
            if evaluate:
                examples = processor.get_dev_examples(input_dir, filename=args.squad_predict_file)
            else:
                examples = processor.get_train_examples(input_dir, filename=args.squad_train_file)
        print("Generated {} examples ".format(len(examples)))
        examples = examples[:10]
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            if not os.path.exists(cached_features_file):
                os.makedirs(os.path.split(cached_features_file)[0])
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

class MyTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_list: str, block_size=128):
        cache_folder = args.cache_folder
        skipped = 0
        for file_path in file_list:
            assert os.path.isfile(file_path)
            directory, filename = os.path.split(file_path)
            cached_features_file = os.path.join(
                cache_folder, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
            )

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                logger.info("Creating features from dataset file at %s", directory)
                if not os.path.isdir(os.path.join(cache_folder)):
                    os.makedirs(os.path.join(cache_folder))
                self.examples = []
                try:
                    with open(file_path, encoding="utf-8") as f:
                        text = f.read()

                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                        self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
                    # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should loook for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    logger.info("Saving features into cached file %s", cached_features_file)
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    logging.info("Skipping {} because of encoding")
                    skipped += 1
        print("{} files are skipped in total ".format(skipped))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])
class MyLineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_list: str, block_size=512):
        self.examples = []
        for file_path in file_list:
            assert os.path.isfile(file_path)
            # Here, we do not cache the features, operating under the assumption
            # that we will soon use fast multithreaded tokenizers from the
            # `tokenizers` repo everywhere =)
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if len(line) > 0]
            example = tokenizer.batch_encode_plus(lines, max_length=block_size)["input_ids"]
            self.examples.extend(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

## Code taken from BERT original source code
## For both next sentence and masked language modeling
class BertPretrainReader():
    def __init__(self,read_dir,tokenizer,flags = None, vocab=None):
        self.FLAGS = flags
        self.read_dir = read_dir
        self.do_whole_word_mask = True
        self.vocab = vocab
        self.tokenizer=tokenizer
        self.dataset = self.create_training_instances(self.read_dir,self.tokenizer)
        if self.vocab:
            self.inv_vocab = {v:k for k,v in self.vocab.items()}
    def create_training_instances(self,input_files, tokenizer, max_seq_length = 128,
                                  dupe_factor  = 5, short_seq_prob = 0.1, masked_lm_prob=0.2,
                                  max_predictions_per_seq = 28, rng = random.Random(random_seed)):
      """Create `TrainingInstance`s from raw text."""
      all_documents = [[]]

      # Input file format:
      # (1) One sentence per line. These should ideally be actual sentences, not
      # entire paragraphs or arbitrary spans of text. (Because we use the
      # sentence boundaries for the "next sentence prediction" task).
      # (2) Blank lines between documents. Document boundaries are needed so
      # that the "next sentence prediction" task doesn't span between documents.
      for input_file in input_files:
        with open(input_file, "r",encoding = 'utf-8') as reader:
          while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
              break
            line = line.strip()

            # Empty lines are used as document delimiters
            if not line:
              all_documents.append([])
            tokens = tokenizer.tokenize(line)
            if tokens:
              all_documents[-1].append(tokens)
      #return all_documents
      # Remove empty documents
      all_documents = [x for x in all_documents if x]
      rng.shuffle(all_documents)

      vocab_words = list(tokenizer.vocab.keys())
      instances = []
      for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
          instances.extend(
              self.create_instances_from_document(
                  all_documents, document_index, max_seq_length, short_seq_prob,
                  masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

      rng.shuffle(instances)
      return instances


    def create_instances_from_document(self,
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
      """Creates `TrainingInstance`s for a single document."""
      document = all_documents[document_index]

      # Account for [CLS], [SEP], [SEP]
      max_num_tokens = max_seq_length - 3

      # We *usually* want to fill up the entire sequence since we are padding
      # to `max_seq_length` anyways, so short sequences are generally wasted
      # computation. However, we *sometimes*
      # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
      # sequences to minimize the mismatch between pre-training and fine-tuning.
      # The `target_seq_length` is just a rough target however, whereas
      # `max_seq_length` is a hard limit.
      target_seq_length = max_num_tokens
      if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

      # We DON'T just concatenate all of the tokens from a document into a long
      # sequence and choose an arbitrary split point because this would make the
      # next sentence prediction task too easy. Instead, we split the input into
      # segments "A" and "B" based on the actual "sentences" provided by the user
      # input.
      instances = []
      current_chunk = []
      current_length = 0
      i = 0
      while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
          if current_chunk:
            # `a_end` is how many segments from `current_chunk` go into the `A`
            # (first) sentence.
            a_end = 1
            if len(current_chunk) >= 2:
              a_end = rng.randint(1, len(current_chunk) - 1)

            tokens_a = []
            for j in range(a_end):
              tokens_a.extend(current_chunk[j])

            tokens_b = []
            # Random next
            is_random_next = False
            if len(current_chunk) == 1 or rng.random() < 0.5:
              is_random_next = True
              target_b_length = target_seq_length - len(tokens_a)

              # This should rarely go for more than one iteration for large
              # corpora. However, just to be careful, we try to make sure that
              # the random document is not the same as the document
              # we're processing.
              for _ in range(10):
                random_document_index = rng.randint(0, len(all_documents) - 1)
                if random_document_index != document_index:
                  break

              random_document = all_documents[random_document_index]
              random_start = rng.randint(0, len(random_document) - 1)
              for j in range(random_start, len(random_document)):
                tokens_b.extend(random_document[j])
                if len(tokens_b) >= target_b_length:
                  break
              # We didn't actually use these segments so we "put them back" so
              # they don't go to waste.
              num_unused_segments = len(current_chunk) - a_end
              i -= num_unused_segments
            # Actual next
            else:
              is_random_next = False
              for j in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[j])
            self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

            assert len(tokens_a) >= 1
            assert len(tokens_b) >= 1

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
              tokens.append(token)
              segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            for token in tokens_b:
              tokens.append(token)
              segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            (tokens, masked_lm_positions,
             masked_lm_labels) = self.create_masked_lm_predictions(
                 tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
            instance = TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=is_random_next,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)
          current_chunk = []
          current_length = 0
        i += 1

      return instances



    def create_masked_lm_predictions(self,tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng):
      """Creates the predictions for the masked LM objective."""

      cand_indexes = []
      for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
          continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (self.do_whole_word_mask and len(cand_indexes) >= 1 and
            token.startswith("##")):
          cand_indexes[-1].append(i)
        else:
          cand_indexes.append([i])

      rng.shuffle(cand_indexes)

      output_tokens = list(tokens)

      num_to_predict = min(max_predictions_per_seq,
                           max(1, int(round(len(tokens) * masked_lm_prob))))

      masked_lms = []
      covered_indexes = set()
      for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
          break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
          continue
        is_any_index_covered = False
        for index in index_set:
          if index in covered_indexes:
            is_any_index_covered = True
            break
        if is_any_index_covered:
          continue
        for index in index_set:
          covered_indexes.add(index)

          masked_token = None
          # 80% of the time, replace with [MASK]
          if rng.random() < 0.8:
            masked_token = "[MASK]"
          else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
              masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
              masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

          output_tokens[index] = masked_token

          masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
      assert len(masked_lms) <= num_to_predict
      masked_lms = sorted(masked_lms, key=lambda x: x.index)

      masked_lm_positions = []
      masked_lm_labels = []
      for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

      return (output_tokens, masked_lm_positions, masked_lm_labels)

    def truncate_seq_pair(self,tokens_a, tokens_b, max_num_tokens, rng):
      """Truncates a pair of sequences to a maximum sequence length."""
      while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
          break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
    def __getitem__(self,ind):
        instance = self.dataset[ind]

        ## if we get them as batch we have to reapply this step
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(instance.tokens))
        mask_label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(instance.masked_lm_labels))
        masked_lm_positions = torch.tensor(instance.masked_lm_positions)
        mask = torch.ones(token_ids.shape,dtype=torch.bool)
        mask[masked_lm_positions] = 0
        mask_labels = token_ids.masked_fill(mask,-100)
        mask_labels[masked_lm_positions] = mask_label_ids
        mask_labels.unsqueeze_(0)
        token_ids.unsqueeze_(0)
        print("Token id shape {} ".format(token_ids.shape))
        next_label = torch.tensor([ 0 if instance.is_random_next  else  1])
        token_type_ids = torch.tensor(instance.segment_ids).unsqueeze(0)
        return token_ids, mask_labels, next_label, token_type_ids



def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

if __name__ == "__main__":
    file_list = ["PMC6961255.txt"]
    args = parse_args()
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    #train_dataset = LineByLineTextDataset(bert_tokenizer,args, file_list)
    train_dataset = MyTextDataset(bert_tokenizer,args,file_list)
    #print("Padding var mii", train_dataset[-1])

    #print(train_dataset[0])
    train_sampler = RandomSampler(train_dataset)
    def collate(examples):
        return pad_sequence(examples, batch_first=True, padding_value=bert_tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=10, collate_fn=collate
    )
    #self.dataset = reader.create_training_instances(file_list,bert_tokenizer)
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print("Padding var mi inside ", batch.shape)
        #print("Batch shape {} ".format(batch.shape))
        #print("First input {} ".format(batch[0]))
        inputs, labels = mask_tokens(batch, bert_tokenizer, args)
        #print(inputs.shape)
