# -*- coding: utf-8 -*-

#Import Required Packages#
import pandas as pd
import keras
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Read In Training Data#
df = pd.read_csv("training.txt",sep="\t",header=None)
df.columns = ["Sentiment","Raw_Text"]

sentiment = df["Sentiment"].values
text = df["Raw_Text"].values

#Split Train/Test#
sentences_train, sentences_test, Y_train, y_test = train_test_split(text,sentiment,random_state=int(input("Random State? " )))

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
x_test  = vectorizer.transform(sentences_test)

#Define Model#
model = Sequential()
model.add(layers.Dense(10, input_dim=X_train.shape[1], activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

#Train Model#
model.fit(X_train, Y_train,
         epochs=100,
         verbose=False,
         validation_data=(x_test,y_test),
         batch_size=10)

#Evaluate Model#
loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.2f}".format(accuracy))
