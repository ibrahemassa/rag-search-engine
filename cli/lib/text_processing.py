import string
import os
from .search_utils import load_stop_words
from nltk.stem import PorterStemmer


def text_preprocessing(text: str) -> str:
    text = str(text)
    text = text.lower()
    puncs = str.maketrans('', '', string.punctuation)
    text = text.translate(puncs)

    return text

def tokenize(text: str) -> list[str]:
    text = str(text)
    stopwords = load_stop_words()
    text = text_preprocessing(text)
    stemmer = PorterStemmer()
    tokens = []
    for token in text.split():
        if token and token not in stopwords:
            tokens.append(stemmer.stem(token))

    return tokens

def tokens_matching(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for t1 in query_tokens:
        for t2 in title_tokens:
            if t1 in t2:
                return True
    return False

