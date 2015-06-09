__author__ = 'Simon Hofmann'

from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation


class Preprocessor:
    __lemmy = None
    __wordnet_tags = ['n', 'v']

    def __init__(self):
        self.__lemmy = WordNetLemmatizer()

    def lemmatize(self, token, tag):
        if tag[0].lower() in self.__wordnet_tags:
            return self.__lemmy.lemmatize(token, tag[0].lower())
        return token

    def remove_punctuation(self, corpus):
        for document in corpus:
            for idx, token in enumerate(document):
                if token[0] in punctuation:
                    del document[idx]
                    
    def remove_stopwords(self, corpus):
        return [[w for w in document if not w in stopwords.words('english')] for document in corpus]

    def process(self, corpus):
        # tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
        self.remove_punctuation(corpus)
        lemmatized_corpus = [[self.lemmatize(token, tag) for token, tag in document] for document in corpus]
        return self.remove_stopwords(lemmatized_corpus)
