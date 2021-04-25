import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

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
    # cleaned = re.sub('[^A-Za-z0-9\s\t]+', '', sentence)
    cleaned = sentence.replace("\ufeff", "")
    # cleaned = remove_stopwords(cleaned)
    return cleaned


def data_reader(dataset_path, encoding='utf-8', skip_unlabeled=False):
    corpus = []
    dataset = open(dataset_path, encoding=encoding).read().split("\n\n")
    i = 0
    for d in tqdm(dataset, desc="Reading the data."):
        if i < 10: print("Instance: ", d.split("\n"))
        words = [x.split()[0] for x in d.split("\n") if len(x.split()) > 0]
        labels = [x.split()[-1] for x in d.split("\n") if len(x.split()) > 0]

        if all([x == "O" for x in labels]):
            continue
        sent = " ".join(words)
        cleaned = data_cleaner(sent)
        if i < 10:
            print(sent, " Cleaned: ", cleaned, labels)
        sent = Sentence(sent, labels, cleaned)
        corpus.append(sent)
        i += 1
    return corpus
