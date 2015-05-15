#!/usr/bin/env python
__author__ = 'Simon Hofmann'

from nb_classifier import NaiveBayesClassifier as NBC
from preprocessor import Preprocessor as Prep


def main():
    nb = NBC()
    p = Prep()

    # Training data
    sports = ["The fans at Staples Center believed it was done too, and Houston Rockets guard Jason Terry could feel it as people in the crowd were yelling for the team in red to go home.",
              "Tom Thibodeau says he expects to be coaching the Chicago Bulls next season despite a league-wide belief that he and the team's front office will decide to part ways.",
              "Kyrie Irving was in the locker room getting his knee checked for additional injury. Kevin Love was back in Cleveland with his arm in a sling. Tristan Thompson was on the bench, his left shoulder wrapped in ice. LeBron James was walking gingerly around the floor, grasping his back after a blow caused a spasm."]

    finance = ["A recovery in the world's largest economy is paving the way for a lift-off in U.S. interest rates, which many analysts expect to take place in September. With the three-month Singapore interbank offered rate, or SIBOR, closely linked to the U.S. Federal funds rate, that liftoff will likely pull the city-state's lending rates higher as well. The Sibor hit a 6-year high of above 1 percent in March and was at 0.8788 per cent on Thursday.",
               "Dealmaking in the United States in 2015 has climbed 48 percent year-on-year to $565.6 billion, the highest level since 2007, following a string of multi-billion dollar acquisitions this week.",
               "Worldwide M&A activity is up 30 percent so far this year compared to the same period in 2014, with $1.4 trillion worth of deals having been struck."]

    documents = [p.process(sports), p.process(finance)]
    nb.train(documents, ["Sports", "Finance"])

    # Test data
    test_finance = "Danaher announced it would buy air and water-filter maker Pall in a $13.8 billion deal, pipeline operator Williams Cos said it would buy affiliate Williams Partners also for about $13.8 billion in stock and Verizon Communications said it would buy AOL in a $4.4 billion deal."
    test_sports = "Following a breakout season that earned him his first NBA MVP trophy, Golden State Warriors point guard Stephen Curry is America's favorite NBA player, according to Public Policy Polling."

    print("Should be: Sports")
    print("Is: " + nb.predict(test_sports))
    print("---")
    print("Should be: Finance")
    print("Is: " + nb.predict(test_finance))

if __name__ == "__main__":
    main()

