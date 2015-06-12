#!/usr/bin/env python
__author__ = 'Simon Hofmann'

from nb_classifier import NaiveBayesClassifier as NBC
from preprocessor import Preprocessor as Prep
from nltk import corpus
from evalutator import Evaluator


def main():
    p = Prep()

    # input("Press Enter to continue...")

    print('Preprocessing...')

    documents = [p.process(corpus.brown.tagged_sents(categories='adventure')),
                 p.process(corpus.brown.tagged_sents(categories='news'))]
    labels = ["fiction", "news"]

    evaluator = Evaluator(documents, labels)
    evaluator.k_fold(5)

    # print('Training...')
    # nb.train(documents, labels)
    # eval.train(documents, labels)
    # nb.info(9)
    #
    # input('Press Enter to continue...')
    #
    # print('Evaluating...')
    #
    # total_hits = 0
    # total_misses = 0
    # total_data = 0
    #
    # for idx, categorie in enumerate(labels):
    #     hits = 0
    #     misses = 0
    #     total_data += len(validation_data[idx])
    #     for item in validation_data[idx]:
    #         print('Should be: ' + categorie)
    #         result = nb.predict(p.process([item]))
    #         print('Is: ' + result)
    #         if result == categorie:
    #             hits += 1
    #             total_hits += 1
    #         else:
    #             misses += 1
    #             total_misses += 1
    #
    #     print('Accuracy: ' + str(hits/len(validation_data[idx]) * 100) + '%')
    #     print('Misses: ' + str(misses/len(validation_data[idx]) * 100) + '%')
    #     input("Press Enter to continue...")
    #
    # print('Total accuracy: ' + str((total_hits/total_data) * 100) + '%')

if __name__ == "__main__":
    main()

