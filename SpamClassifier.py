#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:46:06 2021

@author: asaap
"""

#Importing Dataset

import pandas as pd
messages = pd.read_csv('spam1.csv')

#Data Cleaning and Preprocessing

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating BOW
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values
#y = messages.label

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)

#Training model using NaiveBayes
from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB()
spam_model.fit(X_train, y_train)

y_pred = spam_model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

import pickle
filename = 'spam_classifier.pkl'
pickle.dump(spam_model, open(filename, 'wb'))
pickle.dump(cv, open('count_vect', 'wb'))
