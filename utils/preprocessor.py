import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
import re
import pandas as pd
import json
from datetime import datetime

def convert_to_str(doc):
    return doc if type(doc) == str else ''


tokenizer = TreebankWordTokenizer()
def tokenize(doc):
    return tokenizer.tokenize(doc)


def to_small_letter(word: str):
    return word.lower()


pattern = re.compile("[^a-z.`']")
def except_non_english(pattern, word):
    return pattern.sub('', word)


def trim(word: str):
    return word.strip('.').strip(' ')


def remove_stopwords(words: [str], custom_stopwords: set):
    stop_words = set(stopwords.words('english')) | custom_stopwords
    return [w for w in words if w not in stop_words]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('P'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return ''


n = WordNetLemmatizer()
def lemmatize_with_pos(words):
    words_with_pos = pos_tag(words)

    lemmatized = []
    for w, pos in words_with_pos:
        pos = get_wordnet_pos(pos)
        if pos != '':
            lemmatized.append(n.lemmatize(w, pos))
        else:
            lemmatized.append(w)

    return lemmatized


def map(f, iter):
    return [f(e) for e in iter]

_stopwords = set(json.load(open('dataset/stopwords.json', 'r')))
def preprocessing(doc, stopwords=_stopwords):
    words = tokenize(doc)
    words = map(to_small_letter, words)
    words = [except_non_english(pattern, w) for w in words]
    words = map(trim, words)
    words = [w for w in words if len(w) > 2]
    words = remove_stopwords(words, stopwords)
    words = lemmatize_with_pos(words)
    words = [w for w in words if len(w) > 2]
    return words

