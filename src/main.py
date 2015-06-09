#!/usr/bin/env python
__author__ = 'Simon Hofmann'

from nb_classifier import NaiveBayesClassifier as NBC
from preprocessor import Preprocessor as Prep
from nltk import corpus


def main():
    nb = NBC()
    p = Prep()

    fiction_size = len(corpus.brown.tagged_sents(categories='fiction'))
    science_fiction_size = len(corpus.brown.tagged_sents(categories='science_fiction'))

    # Indices for cross validation
    fiction_training_idx = int((4/5)*fiction_size)
    science_fiction_training_idx = int((4/5)*science_fiction_size)

    # 3399 fiction Datensätze vs. 758 science_fiction Datensätze
    print('Amount of fiction training data: ' + str(fiction_training_idx))
    print('Amount of science-fiction training data: ' + str(science_fiction_training_idx))

    input("Press Enter to continue...")

    # Training data
    fiction_training_data = corpus.brown.tagged_sents(categories='fiction')[0:fiction_training_idx]
    science_fiction_training_data = corpus.brown.tagged_sents(categories='science_fiction')[0:science_fiction_training_idx]

    # Validation data
    fiction_validation_data = corpus.brown.tagged_sents(categories='fiction')[fiction_training_idx:]
    science_fiction_validation_data = corpus.brown.tagged_sents(categories='science_fiction')[science_fiction_training_idx:]

    print('Preprocessing...')
    documents = [p.process(fiction_training_data),
                 p.process(science_fiction_training_data)]
    labels = ["fiction", "science_fiction"]

    print('Training...')
    nb.train(documents, labels)

    print('Evaluating...')
    hits_fiction = 0
    misses_fiction = 0
    for item in fiction_validation_data:
        if nb.predict(p.process([item])) == labels[0]:
            hits_fiction += 1
        else:
            misses_fiction += 1

    # Genauigkeit von 96,94% bei fiction
    print('Accuracy: ' + str(hits_fiction/len(fiction_validation_data)))
    print('Misses: ' + str(misses_fiction/len(fiction_validation_data)))

    input("Press Enter to continue...")

    hits_science_fiction = 0
    misses_science_fiction = 0
    for item in science_fiction_validation_data:
        if nb.predict(p.process([item])) == labels[1]:
            hits_science_fiction += 1
        else:
            misses_science_fiction += 1

    # Gegen Genauigkeit von 7.89% bei science_fiction
    print('Accuracy: ' + str(hits_science_fiction/len(science_fiction_validation_data)))
    print('Misses: ' + str(misses_science_fiction/len(science_fiction_validation_data)))

    print('Total accuracy: ' + str((hits_science_fiction+hits_fiction)/(len(fiction_validation_data)+len(science_fiction_validation_data))))

if __name__ == "__main__":
    main()

