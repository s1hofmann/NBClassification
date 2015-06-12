__author__ = 'Simon Hofmann'

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation

from math import log


class TextClassifier:
    def __init__(self):
        pass
# Insert something useful here


class Preprocessor:
    __wordnet_tags = ['n', 'v']
    __lemmy = None

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
        tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
        self.remove_punctuation(tagged_corpus)
        lemmatized_corpus = [[self.lemmatize(token, tag) for token, tag in document] for document in tagged_corpus]
        return self.remove_stopwords(lemmatized_corpus)


class NaiveBayesClassifier:
    __prior = None
    __conditional = None
    __classes = None
    __vocabulary = None
    __total_docs = 0

    def __init__(self):
        print("Naive bayes classification")
        self.__prior = {}
        self.__classes = {}
        self.__conditional = {}
        self.__vocabulary = set()

    def train(self, documents, label):
        assert (len(documents) == len(label))
        for docs in documents:
            self.__total_docs += len(docs)
            for doc in docs:
                [self.__vocabulary.add(t) for t in doc]

        for idx, docClass in enumerate(label):
            self.__classes[docClass] = 0
            self.__prior[docClass] = len(documents[idx]) / self.__total_docs
            self.__conditional[docClass] = {}
            # Concatenate all documents of current class
            total_text = []
            for docs in documents[idx]:
                [total_text.append(t) for t in docs]

            for token in self.__vocabulary:
                count = total_text.count(token)
                self.__conditional[docClass][token] = (count + 1) / (len(total_text) + len(self.__vocabulary))

    def predict(self, document):
        for docClass in self.__classes:
            self.__classes[docClass] = log(self.__prior[docClass])
            for token in document.split():
                try:
                    self.__classes[docClass] += log(self.__conditional[docClass][token])
                except KeyError:
                    continue

        return max(self.__classes, key=lambda i: self.__classes[i])


def main():
    sports = ["The fans at Staples Center believed it was done too, and Houston Rockets guard Jason Terry could feel it as people in the crowd were yelling for the team in red to go home.",
              "Tom Thibodeau says he expects to be coaching the Chicago Bulls next season despite a league-wide belief that he and the team's front office will decide to part ways.",
              "Kyrie Irving was in the locker room getting his knee checked for additional injury. Kevin Love was back in Cleveland with his arm in a sling. Tristan Thompson was on the bench, his left shoulder wrapped in ice. LeBron James was walking gingerly around the floor, grasping his back after a blow caused a spasm."]
    news = ["A large study of nearly 140,000 people from 17 different countries found a clear and consistent link between grip strength and death from any cause, but especially from heart attack and stroke. It is a better predictor than blood pressure and could be a cheap, quick way for doctors to screen out who needs the most attention.",
            "A controversial new recommendation says women in their 40s at average risk for breast cancer might not need to get an annual mammogram. The guidance is based on a belief that mammograms can lead to overdiagnosis and treatment that, on balance, is worse than waiting until women are older to begin more routine screenings.",
            "The biotech investing world will be glued to their computers Wednesday afternoon. About 5,000 datasets will be released simultaneously at 5 p.m. EDT, ahead of the world's biggest cancer research conference, the American Society of Clinical Oncology meeting."]

    p = Preprocessor()

    documents = [p.process(sports), p.process(news)]
    #china = [["chinese", "bejing"], ["chinese", "shanghai"], ["chinese", "macao"]]
    #japan = [["tokyo", "japan"]]
    #documents = [china, japan]
    #test = "do you know the chinese flag?"

    test_news = "Bancel said his company is trying to help the human body make its own proteins to fight diseases by injecting cells with messenger RNA (mRNA). This contrasts with the traditional method of manufacturing proteins in labs and then giving them to patients in the form of drugs."
    test_sports = "Following a breakout season that earned him his first NBA MVP trophy, Golden State Warriors point guard Stephen Curry is America's favorite NBA player, according to Public Policy Polling."

    nb = NaiveBayesClassifier()

    nb.train(documents, ["Sports", "News"])
    print(nb.predict(test_news))
    print(nb.predict(test_sports))


if __name__ == "__main__":
    main()
