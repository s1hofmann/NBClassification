__author__ = 'Simon Hofmann'

from nb_classifier import NaiveBayesClassifier as NBC
import unittest


class NaiveBayesTest(unittest.TestCase):
    __nb = NBC()

    def test_learning(self):
        china = [["chinese", "bejing"], ["chinese", "shanghai"], ["chinese", "macao"]]
        japan = [["tokyo", "japan"]]
        documents = [china, japan]
        self.__nb.train(documents, ["China", "Japan"])
        self.assertEqual(self.__nb.priors, {'China': 0.75, 'Japan': 0.25})
        self.assertEqual(self.__nb.conditionals, {'China': {'chinese': 0.3333333333333333, 'japan': 0.08333333333333333, 'bejing': 0.16666666666666666, 'macao': 0.16666666666666666, 'shanghai': 0.16666666666666666, 'tokyo': 0.08333333333333333}, 'Japan': {'chinese': 0.125, 'japan': 0.25, 'bejing': 0.125, 'macao': 0.125, 'shanghai': 0.125, 'tokyo': 0.25}})

    def test_prediction(self):
        test = "do you know the chinese flag?"
        self.assertEqual(self.__nb.predict(test), "China")
        self.assertEqual(self.__nb.classes, {'China': -0.2876820724517809, 'Japan': -1.3862943611198906})

if __name__ == "__main__":
    unittest.main()
