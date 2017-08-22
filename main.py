# -*- coding: utf-8 -*-

import utils
import numpy as np
from threading import Thread
import Queue
import argparse

from classifier import Classifier

np.random.seed(1)

# Set-up CLI arg parser
parser = argparse.ArgumentParser(description='NLP final project. This programm will train and test a news classifier'
                                             'using the dataset and algorithm provided as arguments and performing'
                                             '10-fold cross validation. It will ouput the F1 score of each fold.')
# First, set model option
algorithm = parser.add_mutually_exclusive_group(required=True)
algorithm.add_argument('--nb', action='store_true', help="Naive Bayes algorithm")
algorithm.add_argument('--maxent',  action='store_true', help="MaxEnt algorithm")
# Then, set dataset
dataset = parser.add_mutually_exclusive_group(required=True)
dataset.add_argument('--bbc', action='store_true', default=False, help="Use BBC dataset with 5 different classes")
dataset.add_argument('--20n', action='store_true', default=False, help="Use 20newsgroup dataset with 20 classes")
# Set multithread option
parser.add_argument('-t', action='store_true', default=False, help="Select multithread mode")
# Now, obtain arguments
args = parser.parse_args()

# Import module corresponding to the desired model
if args.nb:
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB
    model = "NB"
elif args.maxent:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression
    model = "ME"

# Load dataset
if args.bbc:
    full_set, labels = np.array(utils.load_bbc_dataset())
    dataset = "BBC"
else:
    full_set, labels = np.array(utils.load_20news_dataset())
    dataset = "20news"

# Part the dataset into 10 folds
num_folds = 10
thresholds = utils.cross_validation(full_set, num_folds)

if args.t:
    fscore_top100 = Queue.Queue()
    fscore_feat = Queue.Queue()
    fscore_nofeat = Queue.Queue()
else:
    fscore_top100 = np.zeros(num_folds)
    fscore_feat = np.zeros(num_folds)
    fscore_nofeat = np.zeros(num_folds)

for fold in range(num_folds):
    print "Training and testing fold " + str(fold+1) + "..."
    # Split dataset into train and set based on current fold
    train_set, train_labels, test_set, test_labels = utils.split_set(full_set, labels,
                                                                     thresholds[fold], thresholds[fold+1])

    if args.t:
        t_feat = Thread(target=Classifier(clf(), True, False, args.t).learn_classifier,
                        args=(train_set, train_labels, test_set, test_labels, fscore_feat))
        t_nofeat = Thread(target=Classifier(clf(), False, False, args.t).learn_classifier,
                          args=(train_set, train_labels, test_set, test_labels, fscore_nofeat))
        t_100 = Thread(target=Classifier(clf(), True, True, args.t).learn_classifier,
                       args=(train_set, train_labels, test_set, test_labels, fscore_top100))
        t_feat.start()
        t_nofeat.start()
        t_100.start()
        t_feat.join()
        t_nofeat.join()
        t_100.join()
    else:
        fscore_feat[fold] = Classifier(clf(), True, False, args.t).learn_classifier(train_set, train_labels, test_set, test_labels, fscore_feat)
        fscore_nofeat[fold] = Classifier(clf(), False, False, args.t).learn_classifier(train_set, train_labels, test_set, test_labels, fscore_nofeat)
        fscore_top100[fold] = Classifier(clf(), True, True, args.t).learn_classifier(train_set, train_labels, test_set, test_labels, fscore_top100)

# Convert queue to list if threaded mode is enabled
if args.t:
    fscore_feat = list(fscore_feat.queue)
    fscore_nofeat = list(fscore_nofeat.queue)
    fscore_top100 = list(fscore_top100.queue)

# Print results
print "====== F1 SCORES FOR BAG OF WORD REPRESENTATION ======"
print fscore_nofeat
print
print "====== F1 SCORES FOR FEATURES REPRESENTATION ======"
print fscore_feat
print
print "====== F1 SCORES FOR BIGRAM REPRESENTATION ======"
print fscore_top100
print
