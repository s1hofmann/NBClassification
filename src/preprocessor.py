__author__ = 'Simon Hofmann'

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

    def remove_stopwords(self, corpus):
        return [[w for w in document if not w in stopwords.words('english')] for document in corpus]

    def process(self, corpus):
        # tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
        interpunctuation = punctuation + "''" + "``" + "'" + "`" + "__" + "--"
        lemmatized_corpus = [[self.lemmatize(token, tag).lower() for token, tag in document if token not in interpunctuation] for document in corpus]
        return self.remove_stopwords(lemmatized_corpus)
