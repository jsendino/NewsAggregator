
import numpy as np

from scipy.sparse import csr_matrix

from inspect import getmembers, isfunction, getargspec

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score

import features


class Classifier:
    def __init__(self, classifier, with_feat, with_bigram, is_threaded=False):
        """
        Initialize the classifier instance with important values such as the algorithm or the feature
        representation
        :param classifier: scikit-learn object that is going to be used as classifer
        :param with_feat: boolean variable indicating whether indicator features are being used
        :param with_bigram: boolean variable indicating whether bigrams are being used
        :param is_threaded: boolean variable indicating whether multithread mode is enabled
        """
        self.classifier = classifier
        self.inv_vocab = {}
        self.tfidf_transformer = TfidfTransformer()
        self.count_vect = CountVectorizer()
        if with_bigram:
            self.bigram_vect = CountVectorizer(ngram_range=(2, 2), max_features=100)
            self.tfidf_bigram = TfidfTransformer()

        self.with_feat = with_feat
        self.with_bigram = with_bigram
        self.is_threaded = is_threaded

    def learn_classifier(self, train_ds, train_labels, test_ds, test_labels, queue=[]):
        """
        Train, test and compute the performance of the algorithm (F1 score) with the given train and test set
        :param train_ds: numpy array with the training set
        :param train_labels: numpy array with the training labels
        :param test_ds: numpy array with the testing set
        :param test_labels: numpy array with the testing labels
        :param queue: when threaded mode is enabled, queue where to put the results.
        :return: F1 score if no thread mode is in use
        """
        # Train model
        clf = self.train(train_ds, train_labels)
        # Test it
        predictions = self.test(clf, test_ds)
        # Output results
        if self.is_threaded:
            queue.put(self.accuracy(predictions, test_labels))
        else:
            return self.accuracy(predictions, test_labels)

    def build_feature_matrix(self, dataset):
        """
        Given a new dataset, build the feature matrix with the values of the indicator functions for each
        sample in matrix
        :param dataset: numpy array with all documents in training set
        :return: numpy array with the feature values for each document in training set
        """
        # Create the dictionary of feature functions if it is not created
        if len(features.features_fun_dict) == 0:
            i = 0
            for o in getmembers(features):
                if isfunction(o[1]):
                    features.features_fun_dict[i] = o[1]
                    i += 1
            features.num_features = len(features.features_fun_dict)

        matrix = np.zeros([dataset.shape[0], features.num_features])

        # For each sample in dataset, call every feature function and store its value
        for i in range(dataset.shape[0]):
            for j in range(features.num_features):
                args = getargspec(features.features_fun_dict[j]).args
                if len(args) == 2:
                    matrix[i, j] = features.features_fun_dict[j](dataset[i], self.inv_vocab)
                else:
                    matrix[i, j] = features.features_fun_dict[j](dataset[i])

        # Return sparse matrix with the features (needed by the classifier)
        return csr_matrix(matrix)

    def train(self, train_set, train_labels):
        """
        Train classifier
        :param train_set: numpy array with all documents in training set
        :param train_labels: numpy array with labels in training set
        :return: trained classifier
        """
        # Transform dataset, obtaining the count of every word in vocabulary and performing tfidf conversion
        train_counts = self.count_vect.fit_transform(train_set)
        train_tfidf = self.tfidf_transformer.fit_transform(train_counts)

        # Build inverse vocabulary with all words in dictionary (needed to recover the word from the index
        # in some feature functions)
        self.inv_vocab = {v: k for k, v in self.count_vect.vocabulary_.items()}

        # If using feature representation, obtain the corresponding feature matrix and append
        if self.with_feat:
            matrix = self.build_feature_matrix(train_counts)
            matrix_norm = Normalizer().fit(matrix).transform(matrix)
            train_tfidf = csr_matrix(np.concatenate((train_tfidf.toarray(), matrix_norm.toarray()), axis=1))

        # If using bigram representation, obtain the top 100 bigrams and append
        if self.with_bigram:
            bigram_counts = self.bigram_vect.fit_transform(train_set)
            bigram_tfidf = self.tfidf_bigram.fit_transform(bigram_counts)
            train_tfidf = csr_matrix(np.concatenate((train_tfidf.toarray(), bigram_tfidf.toarray()), axis=1))

        # Return trained classifier
        return self.classifier.fit(train_tfidf, train_labels)

    def test(self, clf, test_set):
        """
        Test classifier
        :param clf: scikit-learn object containing a trained classifier with the desired algorithm
        :param test_set: numpy array with all documents in test set
        :return: numpy array with the predictions
        """
        # Transform test set the same way the training set is transformed
        test_counts = self.count_vect.transform(test_set)
        test_tfidf = self.tfidf_transformer.transform(test_counts)

        # If using feature representation, obtain the corresponding feature matrix and append
        if self.with_feat:
            test_matrix = self.build_feature_matrix(test_counts)
            test_matrix_norm = Normalizer().fit(test_matrix).transform(test_matrix)
            test_tfidf = csr_matrix(np.concatenate((test_tfidf.toarray(), test_matrix_norm.toarray()), axis=1))

        # If using feature representation, obtain the corresponding top 100 bigrams and append
        if self.with_bigram:
            bigram_counts = self.bigram_vect.fit_transform(test_set)
            bigram_tfidf = self.tfidf_bigram.fit_transform(bigram_counts)
            test_tfidf = csr_matrix(np.concatenate((test_tfidf.toarray(), bigram_tfidf.toarray()), axis=1))

        # Return predictions
        return clf.predict(test_tfidf)

    @staticmethod
    def accuracy(predictions, test_labels):
        """
        Compute the desired measure of performance (F1 in this case)
        :param predictions: numpy array with the predictions made by our classifier
        :param test_labels: numpy array with the real labels
        :return: F1 score of the predictions made
        """
        return f1_score(test_labels, predictions, average='micro') * 100

