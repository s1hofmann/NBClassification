__author__ = 'Simon Hofmann'

import unittest
from preprocessor import Preprocessor


class PreprocessorTest(unittest.TestCase):
    __pp = Preprocessor()

    def test_lemmatize(self):
        test_token = 'eaten'
        test_tag = 'VBN'
        self.assertEqual(self.__pp.lemmatize(test_token, test_tag), 'eat')

    def test_remove_punctuation(self):
        test_input = [[('He', 'PRP'),
                       ('ate', 'VBP'),
                       ('all', 'DT'),
                       ('the', 'DT'),
                       ('sandwiches', 'NNS'),
                       ('.', '.')],
                      [('Every', 'DT'),
                       ('sandwich', 'NN'),
                       ('was', 'VBD'),
                       ('eaten', 'VBN'),
                       ('by', 'IN'),
                       ('him', 'PRP'),
                       ('.', '.')]]

        self.__pp.remove_punctuation(test_input)
        self.assertEqual(test_input, [[('He', 'PRP'),
                                       ('ate', 'VBP'),
                                       ('all', 'DT'),
                                       ('the', 'DT'),
                                       ('sandwiches', 'NNS')],
                                      [('Every', 'DT'),
                                       ('sandwich', 'NN'),
                                       ('was', 'VBD'),
                                       ('eaten', 'VBN'),
                                       ('by', 'IN'),
                                       ('him', 'PRP')]])

    def test_remove_stopwords(self):
        test = [["i", "am", "happy", "when", "she", "is"]]
        self.assertEqual(self.__pp.remove_stopwords(test), [['happy']])

    def test_process(self):
        test_corpus = ['He ate all the sandwiches!', 'Every sandwich was eaten by him.']
        self.assertEqual(self.__pp.process(test_corpus), [['He', 'eat', 'sandwich'],
                                                          ['Every', 'sandwich', 'eat']])

if __name__ == "__main__":
    unittest.main()