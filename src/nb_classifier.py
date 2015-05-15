__author__ = 'Simon Hofmann'

from math import log


class NaiveBayesClassifier:
    __prior = None
    __conditional = None
    __classes = None
    __vocabulary = None
    __total_docs = 0

    def __init__(self):
        print("Naive bayes classification")
        self.__prior = {}
        self.__classes = {}
        self.__conditional = {}
        self.__vocabulary = set()

    @property
    def conditionals(self):
        return self.__conditional

    @property
    def classes(self):
        return self.__classes

    @property
    def priors(self):
        return self.__prior

    def train(self, documents, label):
        assert (len(documents) == len(label))
        for docs in documents:
            self.__total_docs += len(docs)
            for doc in docs:
                [self.__vocabulary.add(t) for t in doc]

        for idx, docClass in enumerate(label):
            self.__classes[docClass] = 0
            self.__prior[docClass] = len(documents[idx]) / self.__total_docs
            self.__conditional[docClass] = {}
            # Concatenate all documents of current class
            total_text = []
            for docs in documents[idx]:
                [total_text.append(t) for t in docs]

            for token in self.__vocabulary:
                count = total_text.count(token)
                self.__conditional[docClass][token] = (count + 1) / (len(total_text) + len(self.__vocabulary))

    def predict(self, document):
        for docClass in self.__classes:
            self.__classes[docClass] = log(self.__prior[docClass])
            for token in document.split():
                try:
                    self.__classes[docClass] += log(self.__conditional[docClass][token])
                except KeyError:
                    continue

        print(self.__classes)
        return max(self.__classes, key=lambda i: self.__classes[i])