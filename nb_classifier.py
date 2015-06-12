from operator import itemgetter

__author__ = 'Simon Hofmann'

from math import log


class NaiveBayesClassifier:
    __prior = None
    __conditional = None
    __classes = None
    __vocabulary = None
    __total_docs = 0
    __trained = False

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
        # Build vocabulary
        for docs in documents:
            self.__total_docs += len(docs)
            for doc in docs:
                [self.__vocabulary.add(t) for t in doc]

        # Used to store no. of appearance
        cnt = {}

        for idx, docClass in enumerate(label):
            self.__classes[docClass] = 0
            self.__prior[docClass] = len(documents[idx]) / self.__total_docs
            self.__conditional[docClass] = {}
            total_text = 0
            cnt[idx] = {}
            for docs in documents[idx]:
                for t in docs:
                    total_text += 1
                    try:
                        cnt[idx][t] += 1
                    except KeyError:
                        cnt[idx][t] = 1

            for token in self.__vocabulary:
                count = 0
                try:
                    count = cnt[idx][token]
                except KeyError:
                    pass

                self.__conditional[docClass][token] = (count + 1) / (total_text + len(self.__vocabulary))

        self.__trained = True

    def predict(self, documents):
        for docClass in self.__classes:
            self.__classes[docClass] = log(self.__prior[docClass])
            for document in documents:
                try:
                    for token in document:
                        self.__classes[docClass] += log(self.__conditional[docClass][token])
                except KeyError:
                    continue

        # print(self.__classes)
        return max(self.__classes, key=lambda i: self.__classes[i])

    def info(self, n):
        """
        Prints an overview of n most important vocabulary words for each class.
        """
        if not self.__trained:
            print('This feature is only available after the classifier has been trained!')
            return

        importance_map = {}

        for doc_class in self.__classes:
            importance_map[doc_class] = sorted(self.__conditional[doc_class].items(), key=itemgetter(1), reverse=True)

        for entry in importance_map.keys():
            print('The %d most important tokens for document class: %s' % (n, entry))
            print('*****')
            for item in importance_map[entry][:n]:
                print('Token: %s \t| Importance: %d' % (item[0], item[1]))
