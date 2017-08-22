# -*- coding: utf-8 -*-

import numpy as np
import re
import utils

symbols = [u'*', u'?', u'!', u'¿', u'¡', u'(', u')', u'/', u'&', u'%', u'$', u'€', u'@']

features_fun_dict = {}

num_features = 0


def document_length(doc):
    """
    Returns the number of words in the document
    """
    return sum(doc.data)


def highest_count(doc):
    """
    Returns the number of appearances of the most common words
    """
    return np.max(doc.data)


def num_symbols(doc, inv_vocab):
    """
    Returns the number of letters in the document that are symbols
    """
    count = 0
    for idx in doc.indices:
        word = inv_vocab[idx]
        count += sum([word.count(s) for s in symbols])

    return count


def num_digits(doc, inv_vocab):
    """
    Returns the number of words in the document that are digits
    """
    p = re.compile('\d+(\.\d+)?')

    return sum([1 for idx in doc.indices if p.match(inv_vocab[idx]) is not None])


def has_floats(doc, inv_vocab):
    """
    True if the document contains floats, else False.
    """
    p = re.compile('\d+\.\d+')

    for idx in doc.indices:
        if p.match(inv_vocab[idx]) is not None:
            return 1
    return 0


def has_dates(doc, inv_vocab):
    """
    True if the document has dates, else False.
    """
    for idx in doc.indices:
        if utils.is_date(inv_vocab[idx]):
            return 1
    return 0


def average_word_length(doc, inv_vocab):
    """
    Average number of letters per word in the document
    """
    lengths = []
    for idx in doc.indices:
        lengths.append(len(inv_vocab[idx]))

    return np.mean(lengths)


def num_cap_words(doc, inv_vocab):
    """
    Returns the number of words that are capitalized
    """
    return sum([1 for idx in doc.indices if inv_vocab[idx][0].isupper()])


def num_acronyms(doc, inv_vocab):
    """
    Returns the number of words in document whose letters are all capitalized
    """
    return sum([1 for idx in doc.indices if inv_vocab[idx].isupper()])
