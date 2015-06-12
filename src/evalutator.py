__author__ = 'Simon Hofmann'

from nb_classifier import NaiveBayesClassifier as NBC

class Evaluator:
    __data = None
    __label = None
    __nb = None
    __total_data = []
    __total_labels = []

    def __init__(self, data, label):
        self.__data = data
        self.__label = label
        self.__precision = []

    def k_fold(self, k):
        assert k > 1
        total_length = 0
        for doc in self.__data:
            total_length += len(doc)
        subset_size = int(total_length/k)
        for run in range(0, k):
            print('Run ' + str(run+1))
            nb = NBC()
            testing_data = []
            testing_labels = []
            training_data = []
            training_labels = []
            for idx, d in enumerate(self.__data):
                testing_labels.append(self.__label[idx])
                training_labels.append(self.__label[idx])
                testing_data.append(d[run*subset_size:(run+1)*subset_size])
                training_data.append(d[:run*subset_size] + d[(run+1)*subset_size:])

            nb.train(training_data, training_labels)

            for idx, item in enumerate(testing_labels):
                hits = 0
                misses = 0
                for doc in testing_data[idx]:
                    if nb.predict(doc) == item:
                        hits += 1
                    else:
                        misses += 1
                if len(testing_data[idx]) > 0:
                    self.__precision.append((hits/len(testing_data[idx]), misses/len(testing_data[idx])))

            nb.info(10)
            input('Press Enter to continue...')

        total = 0
        for item in self.__precision:
            print('Precision: %.2d' % (item[0]*100))
            total += item[0]
        total /= len(self.__precision)
        print('Average precision: %.2d' % (total*100))
