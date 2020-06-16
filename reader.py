import tokenization
from transformers import *
# from transformers import squad_convert_examples_to_features
from transformers.tokenization_bert import whitespace_tokenize
# from transformers.
from multiprocessing import Pool, cpu_count
from functools import partial
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

from squad import *

random_seed = 12345
rng = random.Random(random_seed)
log_path = 'read_logger'
# logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')
pretrained_bert_name = 'bert-base-uncased'

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def pubmed_files(root="/home/aakdemir/pubmed/pub/pmc/oa_bulk/"):
    if not os.path.isdir(root):
        return ["PMC6961255.txt", "PMC6958785.txt"]
    # root = "/home/aakdemir/pubmed/pub/pmc/oa_bulk/"
    folder_list = os.listdir(root)
    file_list = []
    names = set()
    for folder in folder_list:
        fold_path = os.path.join(root, folder)
        if os.path.isdir(fold_path):
            for file in os.listdir(fold_path):
                if file.endswith(".txt"):
                    if file in names:
                        print("File with name {} exists in multiple folders")
                    names.add(file)
                    file_list.append(os.path.join(root, folder, file))
    print("Read {} files ".format(len(file_list)))
    return file_list


def my_squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible and not example.is_yes_no:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
            )
        )
    return features


def my_squad_convert_examples_to_features(
        examples, tokenizer, max_seq_length, doc_stride,
        max_query_length, is_training, return_dataset=False, threads=1,
        is_yes_no=False
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    # print(tokenizer)
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
            is_yes_no=is_yes_no
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0

    print("Number of features {} generated from {} examples ".format(len(features), len(examples)))
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        # print(example_features)
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    print("Number of features {} generated from {} examples after for loop".format(len(new_features), len(examples)))
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_squad_bert2tokens = torch.tensor([squad_bert2tokens(f.input_ids, tokenizer) for f in features],
                                             dtype=torch.long)
        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask,
                all_squad_bert2tokens
            )
        else:
            if not is_yes_no:
                all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
                all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_start_positions,
                    all_end_positions,
                    all_cls_index,
                    all_p_mask,
                    all_is_impossible,
                    all_squad_bert2tokens,
                )
            else:
                all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
                ## dummy end position
                all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_start_positions,
                    all_end_positions,
                    all_cls_index,
                    all_p_mask,
                    all_is_impossible,
                    all_squad_bert2tokens,
                )
        print("Just before returning squad f  {} d {}".format(len(features), len(dataset)))
        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                        "is_impossible": ex.is_impossible,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {
                    "start_position": tf.int64,
                    "end_position": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            ),
        )

    return features


def squad_bert2tokens(berttoks, tokenizer):
    berttoks = tokenizer.convert_ids_to_tokens(berttoks)
    # logging.info(berttoks)
    ids = []
    i = 0
    for tok in berttoks:
        ids.append(i)
        # print(tok)
        if not tok.startswith("##"):
            i += 1
    return ids


def squad_load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, yes_no=False, type="factoid"):
    print("YES NO MU {} ".format(yes_no))
    if type == "yesno":
        args.squad_train_file = args.squad_train_yesno_file
        args.squad_predict_file = args.squad_predict_yesno_file
    elif type == "list":
        args.squad_train_file = args.squad_train_list_file
        args.squad_predict_file = args.squad_predict_list_file
    elif type == "factoid":
        args.squad_train_file = args.squad_train_factoid_file
        args.squad_predict_file = args.squad_predict_factoid_file
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cache_folder = args.cache_folder
    # input_dir = args.squad_dir if args.data_dir else "."
    input_dir = args.squad_dir
    if not evaluate:
        input_file_name = os.path.split(args.squad_train_file)[-1].split(".")[0]
    else:
        input_file_name = os.path.split(args.squad_predict_file)[-1].split(".")[0]
    print("READING {} ".format(input_file_name))
    example_size = args.example_num
    cached_features_file = os.path.join(cache_folder,
                                        "cached_{}_{}_{}_{}".format(
                                            input_file_name,
                                            "dev" if evaluate else "train",
                                            list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                            str(args.max_seq_length) + "_" + str(example_size) + ".txt",
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

        if not input_dir and (
                (evaluate and not args.squad_predict_file) or (not evaluate and not args.squad_train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            # processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            processor = SquadV1Processor()
            if evaluate:
                processor.dev_file = args.squad_predict_file
                print("Reading from {} {} ".format(input_dir, args.squad_predict_file))
                examples = processor.get_dev_examples(input_dir,
                                                      filename=args.squad_predict_file,
                                                      only_data=True if args.predict else False,
                                                      )
            else:
                processor.train_file = args.squad_train_file
                examples = processor.get_train_examples(input_dir,
                                                        filename=args.squad_train_file,
                                                        )
        print("Generated {} examples ".format(len(examples)))

        ## Burada datayi kucultmusuz
        if example_size > 0:
            examples = examples[:example_size]
        features, dataset = my_squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
            is_yes_no=yes_no
        )
        print("We have e {} f {} d {} for eval  : {} ".format(len(examples), len(features), len(dataset),
                                                              evaluate))
        if args.local_rank in [-1, 0]:
            if not os.path.exists(cached_features_file):
                folder = os.path.split(cached_features_file)[0]
                if not os.path.exists(folder):
                    os.makedirs(folder)
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
            folder = os.path.split(directory)[-1]
            cached_features_file = os.path.join(
                cache_folder, args.model_type + "_cached_lm_" + str(block_size) + "_" + folder + "_" + filename
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

                    for i in range(0, len(tokenized_text) - block_size + 1,
                                   block_size):  # Truncate in block of block_size
                        self.examples.append(
                            tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
                    # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should loook for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    logger.info("Saving features into cached file %s", cached_features_file)
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:

                    logging.info("Skipping {} because of encoding".format(file_path))
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
    def __init__(self, read_dir, tokenizer, flags=None, vocab=None):
        self.FLAGS = flags
        self.read_dir = read_dir
        self.do_whole_word_mask = True
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.dataset = self.create_training_instances(self.read_dir, self.tokenizer)
        if self.vocab:
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def create_training_instances(self, input_files, tokenizer, max_seq_length=128,
                                  dupe_factor=5, short_seq_prob=0.1, masked_lm_prob=0.2,
                                  max_predictions_per_seq=28, rng=random.Random(random_seed)):
        """Create `TrainingInstance`s from raw text."""
        all_documents = [[]]

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        for input_file in input_files:
            with open(input_file, "r", encoding='utf-8') as reader:
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
        # return all_documents
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

    def create_masked_lm_predictions(self, tokens, masked_lm_prob,
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

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
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

    def __getitem__(self, ind):
        instance = self.dataset[ind]

        ## if we get them as batch we have to reapply this step
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(instance.tokens))
        mask_label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(instance.masked_lm_labels))
        masked_lm_positions = torch.tensor(instance.masked_lm_positions)
        mask = torch.ones(token_ids.shape, dtype=torch.bool)
        mask[masked_lm_positions] = 0
        mask_labels = token_ids.masked_fill(mask, -100)
        mask_labels[masked_lm_positions] = mask_label_ids
        mask_labels.unsqueeze_(0)
        token_ids.unsqueeze_(0)
        print("Token id shape {} ".format(token_ids.shape))
        next_label = torch.tensor([0 if instance.is_random_next else 1])
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
    # train_dataset = LineByLineTextDataset(bert_tokenizer,args, file_list)
    train_dataset = MyTextDataset(bert_tokenizer, args, file_list)
    # print("Padding var mii", train_dataset[-1])

    # print(train_dataset[0])
    train_sampler = RandomSampler(train_dataset)


    def collate(examples):
        return pad_sequence(examples, batch_first=True, padding_value=bert_tokenizer.pad_token_id)


    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=10, collate_fn=collate
    )
    # self.dataset = reader.create_training_instances(file_list,bert_tokenizer)
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print("Padding var mi inside ", batch.shape)
        # print("Batch shape {} ".format(batch.shape))
        # print("First input {} ".format(batch[0]))
        inputs, labels = mask_tokens(batch, bert_tokenizer, args)
        # print(inputs.shape)
