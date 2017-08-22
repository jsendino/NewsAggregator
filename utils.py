from nltk import FreqDist

from collections import defaultdict

import os
import numpy as np

import sklearn
from sklearn.datasets import fetch_20newsgroups

from nltk.stem.porter import PorterStemmer
import nltk

from dateutil.parser import parse

unknown_token = "<UNK>"  # unknown word token.
labels = {"business": 0, "entertainment": 1, "politics": 2, "sport": 3, "tech": 4}


def is_date(string):
    try:
        parse(string)
        return True
    except:
        return False


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems


def simple_token(text):
    return text.split()


def read_dataset(folder):
    docs = sklearn.datasets.load_files(folder, encoding="ISO-8859-1")

    return docs.data, docs.target


def load_bbc_dataset():
    return read_dataset("bbc")


def load_20news_dataset(categories=None):
    dataset = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
    return dataset.data, dataset.target


def split_set(ds, classes, fold_init, fold_end):
    # Index all the samples randomly
    num_samples = ds.shape[0]

    # Part that index into train and test samples
    test_idx = range(fold_init, fold_end)
    train_idx = np.setdiff1d(range(num_samples), test_idx)

    return ds[train_idx], classes[train_idx], ds[test_idx], classes[test_idx]


def cross_validation(train_set, num_folds):
    # Get number of samples in each fold as the total number of samples divided by the number of folds
    num_samples = train_set.shape[0]
    samples_per_fold = int(num_samples / num_folds)

    # Compute which sample is the threshold of each fold (the first sample in each fold)
    subset_thresholds = [i*samples_per_fold for i in range(num_folds)]

    # Get the number of samples that, due to the fact that num_samples may not be divisible by num_folds,
    # are not assigned to any fold
    diff = num_samples - samples_per_fold * num_folds

    # If there are not assigned samples, assigned them one to each fold.
    if diff != 0:
        for fold in range(diff):
            subset_thresholds[fold+1:] = [threshold + 1 for threshold in subset_thresholds[fold+1:]]

    return subset_thresholds + [num_samples]