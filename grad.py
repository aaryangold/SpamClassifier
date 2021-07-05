#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:48:12 2021

@author: asaap
"""

import gradio as gr
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
messages = pd.read_csv('spam1.csv')

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


def spam_classifier(message):
    ps = PorterStemmer()
    corpus1 = []
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus1.append(review)
    
    X = cv.transform(corpus1).toarray()
    
    prediction = spam_model.predict(X)
    if prediction==1:
        return 'SPAM!'
    else:
        return 'HAM.'

iface = gr.Interface(
  fn=spam_classifier, 
  inputs=gr.inputs.Textbox(lines=2, placeholder="Paste Your Message Here..."), 
  outputs="text")
iface.launch()
