from typing import List
import re
import pandas as pd
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text: str) -> str:
    """Canonical clean function"""
    if pd.isnull(text):  # WARN: returns 'True' for None object, but we don't expect None objects anyways
        text = ''

    if not isinstance(text, str):
        text = str(text)

    text = text \
        .replace('/s/', '') \
        .replace('.%', '') \
        .replace('%', '') \
        .replace('$', '') \
        .replace('_', '') \
        .replace('--', '') \
        .replace('//', '') \
        .replace('{', '') \
        .replace('}', '')

    text = re.sub('[0-9]+', '', text)  # metis 1.x removing numbers for divs
    text = re.sub('[*]+', '', text)
    text = re.sub('¿', '', text)
    text = re.sub('¡', '', text)

    text = text.strip()
    return text


def tokenize_sent(text):
    return nltk.sent_tokenize(text)


def sent_tokenize_pad(text, max_word_len, tokenizer):
    encoded_doc = tokenizer.texts_to_sequences(text)
    padded_doc = pad_sequences(encoded_doc, maxlen=max_word_len)
    return padded_doc
