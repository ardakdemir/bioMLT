import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize


class Sentence:
    def __init__(self, sent, labels, preprocessed):
        self.sent = sent
        self.labels = labels
        self.preprocessed = preprocessed
        self.words = self.sent.split(" ")


def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    no_stop_sent = " ".join(tokens_without_sw)
    return no_stop_sent


def data_cleaner(sentence):
    cleaned = re.sub('[^A-Za-z0-9\s\t]+', '', sentence)
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = remove_stopwords(cleaned)
    return cleaned


def data_reader(dataset_path, encoding='utf-8', skip_unlabeled=False):
    corpus = []
    dataset = open(dataset_path, encoding=encoding).read().split("\n\n")
    for d in dataset:
        words, labels = zip(*[x.split("\t") for x in d.split("\n")])
        if all([x == "O" for x in labels]):
            continue
        sent = " ".join(words)
        cleaned = data_cleaner(sent)
        sent = Sentence(sent, labels, cleaned)
        corpus.append(sent)
    return corpus
