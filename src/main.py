#!/usr/bin/env python
__author__ = 'Simon Hofmann'

from nltk import corpus
from evalutator import Evaluator

def main():
    d1 = corpus.brown.tagged_sents(categories='humor')
    d2 = corpus.brown.tagged_sents(categories='learned')
    documents = [d1, d2]
    labels = ["l1", "l2"]

    evaluator = Evaluator(documents, labels, True, 5)
    evaluator.k_fold(5)

if __name__ == "__main__":
    main()

