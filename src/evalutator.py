__author__ = 'Simon Hofmann'

from nb_classifier import NaiveBayesClassifier as NBC
from preprocessor import Preprocessor as Prep

class Evaluator:
    __data = []
    __label = None
    __nb = None
    __prep = None
    __total_data = []
    __total_labels = []
    __verbose = False
    __level = 0

    def __init__(self, data, label, verbose=False, verbosity_level = 5):
        print('Preprocessing data...')
        self.__prep = Prep()
        for item in data:
            self.__data.append(self.__prep.process(item))
        self.__label = label
        self.__precision = []
        self.__verbose = verbose
        self.__level = verbosity_level

    def k_fold(self, k):
        assert k > 1
        print('Starting ' + str(k) + '-fold cross-validation.')
        input('Press Enter to continue...')
        for run in range(0, k):
            print('Run ' + str(run+1))
            nb = NBC()
            testing_data = []
            training_data = []
            for idx, d in enumerate(self.__data):
                subset_size = int(len(d)/k)
                testing_data.append(d[run*subset_size:(run+1)*subset_size])
                training_data.append(d[:run*subset_size] + d[(run+1)*subset_size:])

            nb.train(training_data, self.__label)

            hits = 0
            misses = 0
            for idx, item in enumerate(self.__label):
                for doc in testing_data[idx]:
                    if nb.predict(doc) == item:
                        hits += 1
                    else:
                        misses += 1

            total_length = 0
            for item in testing_data:
                total_length += len(item)

            self.__precision.append((hits/total_length, misses/total_length))

            if self.__verbose:
                nb.info(self.__level)
                input('Press Enter to continue...')

        total = 0
        for item in self.__precision:
            print('Precision: %.2d' % (item[0]*100))
            total += item[0]
        total /= len(self.__precision)
        print('Average precision: %.2d' % (total*100))
