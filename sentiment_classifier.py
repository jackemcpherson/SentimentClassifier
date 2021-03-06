# -*- coding: utf-8 -*-

#Import Required Packages#

import pandas as pd
import textblob as tb
from textblob.classifiers import NaiveBayesClassifier
import nltk
nltk.download('punkt')
nltk.download('brown')

#Version Check#

print(pd.__version__)
print(tb.__version__)
print(nltk.__version__)

#Data Cleanse and Shuffle#

train = pd.read_csv("training.txt", sep="\t", header=None)
train = train.sample(frac=1,random_state=int(input("Random State? " )))
train.columns = ["Sentiment","Raw_Text"]
train_data = list(zip(train["Raw_Text"],train["Sentiment"]))

#Train Classifier#

cl = NaiveBayesClassifier(train_data[:1000])

#Test Classifier#

train["Guess"] = train["Raw_Text"].apply(cl.classify)

#Results#

Accuracy = round((train[train["Sentiment"] == train["Guess"]].size/train.size)*100, 2)
print(f"\nAccuracy: {Accuracy}%\n")
print(cl.show_informative_features())
